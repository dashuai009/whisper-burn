use burn::{
    config::Config,
    module::{Module, Param, ParamId},
    nn::{
        self,
        conv::{Conv1d, Conv1dConfig},
        Embedding, EmbeddingConfig, Linear, PaddingConfig1d,
    },
    tensor::{activation::softmax, backend::Backend, Int, Tensor},
};

#[derive(Config, Debug)]
pub struct WhisperConfig {
    audio_encoder_config: AudioEncoderConfig,
    text_decoder_config: TextDecoderConfig,
}

impl WhisperConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Whisper<B> {
        let n_audio_state = self.audio_encoder_config.n_audio_state;
        let n_text_state = self.text_decoder_config.n_text_state;

        assert_eq!(
            n_audio_state, n_text_state,
            "Audio encoder state size {} must be equal to text decoder state size {}.",
            n_audio_state, n_text_state
        );

        let encoder = self.audio_encoder_config.init(device);
        let decoder = self.text_decoder_config.init(device);

        Whisper { encoder, decoder }
    }
}

#[derive(Module, Debug)]
pub struct Whisper<B: Backend> {
    encoder: AudioEncoder<B>,
    decoder: TextDecoder<B>,
}

impl<B: Backend> Whisper<B> {
    pub fn forward(&self, mel: Tensor<B, 3>, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.decoder.forward(tokens, self.encoder.forward(mel))
    }

    pub fn forward_encoder(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.forward(mel)
    }

    pub fn forward_decoder(
        &self,
        tokens: Tensor<B, 2, Int>,
        encoder_output: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.decoder.forward(tokens, encoder_output)
    }

    pub fn encoder_ctx_size(&self) -> usize {
        self.encoder.ctx_size()
    }

    pub fn decoder_ctx_size(&self) -> usize {
        self.decoder.ctx_size()
    }
}

#[derive(Config, Debug)]
pub struct TextDecoderConfig {
    n_vocab: usize,
    n_text_ctx: usize,
    n_text_state: usize,
    n_text_head: usize,
    n_text_layer: usize,
}

impl TextDecoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TextDecoder<B> {
        let token_embedding = EmbeddingConfig::new(self.n_vocab, self.n_text_state).init(device);
        let positional_embedding = Param::initialized(
            ParamId::new(),
            Tensor::<B, 2>::empty([self.n_text_ctx, self.n_text_state], device),
        );
        let blocks: Vec<_> = (0..self.n_text_layer)
            .into_iter()
            .map(|_| {
                ResidualDecoderAttentionBlockConfig::new(self.n_text_state, self.n_text_head)
                    .init(device)
            })
            .collect();
        let ln = nn::LayerNormConfig::new(self.n_text_state).init(device);

        let mask = Tensor::<B, 2>::full(
            [self.n_text_ctx, self.n_text_ctx],
            f32::NEG_INFINITY,
            device,
        )
        .triu(1)
        .into();

        let n_vocab = self.n_vocab;
        let n_text_ctx = self.n_text_ctx;

        TextDecoder {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
            n_vocab,
            n_text_ctx,
        }
    }
}

#[derive(Module, Debug)]
pub struct TextDecoder<B: Backend> {
    token_embedding: Embedding<B>,
    positional_embedding: Param<Tensor<B, 2>>,
    blocks: Vec<ResidualDecoderAttentionBlock<B>>,
    ln: nn::LayerNorm<B>,
    mask: Tensor<B, 2>,
    n_vocab: usize,
    n_text_ctx: usize,
}

impl<B: Backend> TextDecoder<B> {
    fn forward(&self, x: Tensor<B, 2, Int>, xa: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_n_batch, seq_len] = x.dims();

        assert!(
            seq_len <= self.n_text_ctx,
            "Token sequence length {} must not exceed {}.",
            seq_len,
            self.n_text_ctx
        );

        let x = self.token_embedding.forward(x)
            + self
                .positional_embedding
                .val()
                .slice([0..seq_len])
                .unsqueeze::<3>();

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x, xa.clone(), &self.mask);
        }

        let x = self.ln.forward(x);
        return x.matmul(
            self.token_embedding
                .weight
                .val()
                .transpose()
                .unsqueeze::<3>(),
        );
    }

    fn ctx_size(&self) -> usize {
        self.n_text_ctx
    }
}

// def sinusoids(length, channels, max_timescale=10000):
//     """Returns sinusoids for positional embedding"""
//     assert channels % 2 == 0
//     log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
//     inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
//     scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
//     return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
//

fn sinusoids<B: Backend>(length: usize, channels: usize, device: &B::Device) -> Tensor<B, 2> {
    assert_eq!(channels % 2, 0, "");
    let max_timescale = 10000.0;
    let log_timescale_increment = f32::ln(max_timescale) / ((channels / 2 - 1) as f32);
    let inv_timescales = Tensor::exp(
        Tensor::arange(0..((channels / 2) as i64), device)
            .float()
            .mul_scalar(-log_timescale_increment),
    );
    let scaled_time = Tensor::arange(0..(length as i64), device)
        .reshape([length, 1])
        .float()
        .matmul(inv_timescales.reshape([1, channels / 2]));
    return Tensor::cat(vec![scaled_time.clone().sin(), scaled_time.cos()], 1);
}

#[derive(Config, Debug)]
pub struct AudioEncoderConfig {
    n_mels: usize,
    n_audio_ctx: usize,
    n_audio_state: usize,
    n_audio_head: usize,
    n_audio_layer: usize,
}

impl AudioEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AudioEncoder<B> {
        println!("begin init audio encoder");
        let conv1 = Conv1dConfig::new(self.n_mels, self.n_audio_state, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .init(device);
        let conv2 = Conv1dConfig::new(self.n_audio_state, self.n_audio_state, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_stride(2)
            .init(device);
        let blocks: Vec<_> = (0..self.n_audio_layer)
            .into_iter()
            .map(|_| {
                ResidualEncoderAttentionBlockConfig::new(self.n_audio_state, self.n_audio_head)
                    .init(device)
            })
            .collect();
        let ln_post = nn::LayerNormConfig::new(self.n_audio_state).init(device);
        let positional_embedding = sinusoids(self.n_audio_ctx, self.n_audio_state, device);
        let n_mels = self.n_mels;
        let n_audio_ctx = self.n_audio_ctx;

        AudioEncoder {
            conv1,
            conv2,
            blocks,
            ln_post,
            positional_embedding,
            n_mels,
            n_audio_ctx,
        }
    }
}

#[derive(Module, Debug)]
pub struct AudioEncoder<B: Backend> {
    conv1: Conv1d<B>,
    conv2: Conv1d<B>,
    blocks: Vec<ResidualEncoderAttentionBlock<B>>,
    ln_post: nn::LayerNorm<B>,
    positional_embedding: Tensor<B, 2>,
    n_mels: usize,
    n_audio_ctx: usize,
}

impl<B: Backend> AudioEncoder<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, n_mels, _n_ctx] = x.dims();

        assert_eq!(
            n_mels, self.n_mels,
            "Audio mel spectrum size must be {}.",
            self.n_mels
        );
        // assert!(
        //     n_ctx <= self.n_audio_ctx,
        //     "Audio length {} cannot exceed {}.",
        //     n_ctx,
        //     self.n_audio_ctx
        // );

        let x = nn::Gelu::new().forward(self.conv1.forward(x));
        let x = nn::Gelu::new().forward(self.conv2.forward(x));
        let x = x.permute([0, 2, 1]);

        let k = x.dims()[0];
        let x = x + self
            .positional_embedding
            .clone()
            .unsqueeze::<3>()
            .repeat(0, k);

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x);
        }

        return self.ln_post.forward(x);
    }

    fn ctx_size(&self) -> usize {
        self.n_audio_ctx
    }
}

#[derive(Config)]
pub struct ResidualEncoderAttentionBlockConfig {
    n_state: usize,
    n_head: usize,
}

impl ResidualEncoderAttentionBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResidualEncoderAttentionBlock<B> {
        let attn = MultiHeadSelfAttentionConfig::new(self.n_state, self.n_head).init(device);
        let attn_ln = nn::LayerNormConfig::new(self.n_state).init(device);

        let mlp0 = nn::LinearConfig::new(self.n_state, 4 * self.n_state).init(device);
        let mlp2 = nn::LinearConfig::new(4 * self.n_state, self.n_state).init(device);

        let mlp_ln = nn::LayerNormConfig::new(self.n_state).init(device);

        ResidualEncoderAttentionBlock {
            attn,
            attn_ln,
            mlp0,
            mlp2,
            mlp_ln,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualEncoderAttentionBlock<B: Backend> {
    attn: MultiHeadSelfAttention<B>,
    attn_ln: nn::LayerNorm<B>,
    mlp0: Linear<B>,
    mlp2: Linear<B>,
    mlp_ln: nn::LayerNorm<B>,
}

impl<B: Backend> ResidualEncoderAttentionBlock<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.attn_ln.forward(x), None);

        let z = self.mlp_ln.forward(x.clone());

        let z = self.mlp0.forward(z);
        let z = nn::Gelu::new().forward(z);
        let z = self.mlp2.forward(z);

        let x = x + z;
        return x;
    }
}

#[derive(Config)]
pub struct ResidualDecoderAttentionBlockConfig {
    n_state: usize,
    n_head: usize,
}

impl ResidualDecoderAttentionBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResidualDecoderAttentionBlock<B> {
        let attn = MultiHeadSelfAttentionConfig::new(self.n_state, self.n_head).init(device);
        let attn_ln = nn::LayerNormConfig::new(self.n_state).init(device);

        let cross_attn = MultiHeadCrossAttentionConfig::new(self.n_state, self.n_head).init(device);
        let cross_attn_ln = nn::LayerNormConfig::new(self.n_state).init(device);

        let mlp0 = nn::LinearConfig::new(self.n_state, 4 * self.n_state).init(device);
        let mlp2 = nn::LinearConfig::new(4 * self.n_state, self.n_state).init(device);

        let mlp_ln = nn::LayerNormConfig::new(self.n_state).init(device);

        ResidualDecoderAttentionBlock {
            attn,
            attn_ln,
            cross_attn,
            cross_attn_ln,
            mlp0,
            mlp2,
            mlp_ln,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualDecoderAttentionBlock<B: Backend> {
    attn: MultiHeadSelfAttention<B>,
    attn_ln: nn::LayerNorm<B>,
    cross_attn: MultiHeadCrossAttention<B>,
    cross_attn_ln: nn::LayerNorm<B>,
    mlp0: Linear<B>,
    mlp2: Linear<B>,
    mlp_ln: nn::LayerNorm<B>,
}

impl<B: Backend> ResidualDecoderAttentionBlock<B> {
    fn forward(&self, x: Tensor<B, 3>, xa: Tensor<B, 3>, mask: &Tensor<B, 2>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.attn_ln.forward(x), Some(mask));
        let x = x.clone() + self.cross_attn.forward(self.cross_attn_ln.forward(x), xa);

        let z = self.mlp_ln.forward(x.clone());

        let z = self.mlp0.forward(z);
        let z = nn::Gelu::new().forward(z);
        let z = self.mlp2.forward(z);

        let x = x + z;
        return x;
    }
}

#[derive(Config)]
pub struct MultiHeadSelfAttentionConfig {
    n_state: usize,
    n_head: usize,
}

impl MultiHeadSelfAttentionConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadSelfAttention<B> {
        assert!(
            self.n_state % self.n_head == 0,
            "State size {} must be a multiple of head size {}",
            self.n_state,
            self.n_head
        );

        let n_head = self.n_head;
        let query = nn::LinearConfig::new(self.n_state, self.n_state).init(device);
        let key = nn::LinearConfig::new(self.n_state, self.n_state)
            .with_bias(false)
            .init(device);
        let value = nn::LinearConfig::new(self.n_state, self.n_state).init(device);
        let out = nn::LinearConfig::new(self.n_state, self.n_state).init(device);

        MultiHeadSelfAttention {
            n_head,
            query,
            key,
            value,
            out,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadSelfAttention<B: Backend> {
    n_head: usize,
    query: nn::Linear<B>,
    key: nn::Linear<B>,
    value: nn::Linear<B>,
    out: nn::Linear<B>,
}

impl<B: Backend> MultiHeadSelfAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, mask: Option<&Tensor<B, 2>>) -> Tensor<B, 3> {
        let q = self.query.forward(x.clone());
        let k = self.key.forward(x.clone());
        let v = self.value.forward(x);

        let wv = qkv_attention(q, k, v, mask, self.n_head);

        return self.out.forward(wv);
    }
}

#[derive(Config)]
pub struct MultiHeadCrossAttentionConfig {
    n_state: usize,
    n_head: usize,
}

impl MultiHeadCrossAttentionConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> MultiHeadCrossAttention<B> {
        assert!(
            self.n_state % self.n_head == 0,
            "State size {} must be a multiple of head size {}",
            self.n_state,
            self.n_head
        );

        let n_head = self.n_head;
        let query = nn::LinearConfig::new(self.n_state, self.n_state).init(device);
        let key = nn::LinearConfig::new(self.n_state, self.n_state)
            .with_bias(false)
            .init(device);
        let value = nn::LinearConfig::new(self.n_state, self.n_state).init(device);
        let out = nn::LinearConfig::new(self.n_state, self.n_state).init(device);

        MultiHeadCrossAttention {
            n_head,
            query,
            key,
            value,
            out,
        }
    }
}

#[derive(Module, Debug)]
pub struct MultiHeadCrossAttention<B: Backend> {
    n_head: usize,
    query: nn::Linear<B>,
    key: nn::Linear<B>,
    value: nn::Linear<B>,
    out: nn::Linear<B>,
}

impl<B: Backend> MultiHeadCrossAttention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, xa: Tensor<B, 3>) -> Tensor<B, 3> {
        let q = self.query.forward(x);
        let k = self.key.forward(xa.clone());
        let v = self.value.forward(xa);

        let wv = qkv_attention(q, k, v, None, self.n_head);

        return self.out.forward(wv);
    }
}

pub fn qkv_attention<B: Backend>(
    q: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
    mask: Option<&Tensor<B, 2>>,
    n_head: usize,
) -> Tensor<B, 3> {
    let [n_batch, n_qctx, n_state] = q.dims();
    let [_, n_ctx, _] = k.dims();

    let scale = (n_state as f64 / n_head as f64).powf(-0.25);
    let n_hstate = n_state / n_head;

    let q = q
        .reshape([n_batch, n_qctx, n_head, n_hstate])
        .permute([0, 2, 1, 3])
        * scale;
    let k = k
        .reshape([n_batch, n_ctx, n_head, n_hstate])
        .permute([0, 2, 3, 1])
        * scale;
    let v = v
        .reshape([n_batch, n_ctx, n_head, n_hstate])
        .permute([0, 2, 1, 3]);

    let qk = q.matmul(k);

    // apply mask
    let qk = if let Some(mask) = mask {
        qk + mask.clone().slice([0..n_qctx, 0..n_ctx]).unsqueeze::<4>()
    } else {
        qk
    };

    // normalize value weightings
    let w = softmax(qk, 3);
    let o = w.matmul(v).permute([0, 2, 1, 3]).flatten(2, 3);

    return o;
}

#[cfg(test)]
mod test {
    use burn::prelude::Tensor;

    #[test]
    fn test_triu() {
        cfg_if::cfg_if! {
            if #[cfg(feature = "wgpu-backend")] {
                type CurBackend = burn_wgpu::Wgpu<burn_wgpu::AutoGraphicsApi, f32, i32>;
                let device = burn_wgpu::WgpuDevice::BestAvailable;
            } else if #[cfg(feature = "torch-backend")] {
                type CurBackend = LibTorch<f32>;
                let device = LibTorchDevice::Cuda(0);
            }
        }
        let mask = Tensor::<CurBackend, 2>::full([10, 8], f32::NEG_INFINITY, &device).triu(1);
        println!("{mask}");
    }
}
