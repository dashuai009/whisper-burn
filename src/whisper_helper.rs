use std::collections::HashMap;
use std::rc::Rc;

use burn::config::ConfigError;
use burn::prelude::{Backend, Bool, Device, Int, Tensor};
use burn::record::Recorder;
use burn::tensor::activation::softmax;
use hf_hub;
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use crate::audio::{log_mel_spectrogram, N_SAMPLES};

use crate::decoding::{
    DecodingOptions, GreedyDecoder, logit_filter::{LogitFilter, SuppressBlank}, sequence_ranker::{SequenceRanker, TakeFirstGroup},
    TokenDecoder,
    UserSuppressTokens,
};
use crate::model::{Whisper, WhisperConfig};
use crate::token::{Gpt2Tokenizer, Language, SpecialToken};

use crate::model_config::MODEL_CONFIG;

/// the kind of model
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize, Hash)]
pub enum WhichModel {
    Tiny,
    // #[value(name = "tiny.en")]
    TinyEn,
    #[default]
    Base,
    // #[value(name = "base.en")]
    BaseEn,
    Small,
    // #[value(name = "small.en")]
    SmallEn,
    Medium,
    // #[value(name = "medium.en")]
    MediumEn,
    LargeV1,
    LargeV2,
    LargeV3,
    // #[value(name = "distil-medium.en")]
    DistilMediumEn,
    // #[value(name = "distil-large-v2")]
    DistilLargeV2,
}

impl WhichModel {
    /// model.dims.n_vocab >= 51865

    fn is_multilingual(&self) -> bool {
        match self {
            Self::Tiny
            | Self::Base
            | Self::Small
            | Self::Medium
            | Self::LargeV1
            | Self::LargeV2
            | Self::LargeV3
            | Self::DistilLargeV2 => true,

            Self::TinyEn | Self::BaseEn | Self::SmallEn | Self::MediumEn | Self::DistilMediumEn => {
                false
            }
        }
    }


    pub fn model_and_revision(&self) -> (&'static str, &'static str) {
        match self {
            Self::Tiny => ("openai/whisper-tiny", "main"),
            Self::TinyEn => ("openai/whisper-tiny.en", "refs/pr/15"),
            Self::Base => ("openai/whisper-base", "refs/pr/22"),
            Self::BaseEn => ("openai/whisper-base.en", "refs/pr/13"),
            Self::Small => ("openai/whisper-small", "main"),
            Self::SmallEn => ("openai/whisper-small.en", "refs/pr/10"),
            Self::Medium => ("openai/whisper-medium", "main"),
            Self::MediumEn => ("openai/whisper-medium.en", "main"),
            Self::LargeV1 => ("openai/whisper-large", "refs/pr/36"),
            Self::LargeV2 => ("openai/whisper-large-v2", "refs/pr/57"),
            Self::LargeV3 => ("openai/whisper-large-v3", "main"),
            Self::DistilMediumEn => ("distil-whisper/distil-medium.en", "main"),
            Self::DistilLargeV2 => ("distil-whisper/distil-large-v2", "main"),
        }
    }

    fn load_config(&self) -> Result<WhisperConfig, ConfigError> {
        let json_str = match self {
            WhichModel::Tiny => { MODEL_CONFIG[0] }
            WhichModel::TinyEn => { MODEL_CONFIG[1] }
            WhichModel::Base => { MODEL_CONFIG[2] }
            WhichModel::BaseEn => { MODEL_CONFIG[3] }
            WhichModel::Small => { MODEL_CONFIG[4] }
            WhichModel::SmallEn => { MODEL_CONFIG[5] }
            WhichModel::Medium => { MODEL_CONFIG[6] }
            WhichModel::MediumEn => { MODEL_CONFIG[7] }
            WhichModel::LargeV1 => { MODEL_CONFIG[8] }
            WhichModel::LargeV2 => { MODEL_CONFIG[8] }
            WhichModel::LargeV3 => { MODEL_CONFIG[8] }
            WhichModel::DistilMediumEn => { MODEL_CONFIG[8] }
            WhichModel::DistilLargeV2 => { MODEL_CONFIG[8] }
        };
        serde_json::from_str(json_str).map_err(|err| ConfigError::InvalidFormat(format!("{err}")))
    }
}


#[derive(Clone)]
struct _BeamSearchToken {
    _token: usize,
    _log_prob: f64,
}


#[derive(Debug)]
pub struct DecodingResult<B: Backend> {
    pub audio_features: Tensor<B, 2>,
    pub language: Language,
    pub language_probs: HashMap<String, f32>,
    //  Optional[Dict[str, float]] = None
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f32,
    pub no_speech_prob: f32,
    pub temperature: f32,
    pub compression_ratio: f32,
}

impl<B: Backend> DecodingResult<B> {
    fn new(
        audio_features: Tensor<B, 2>,
        language: Language,
        language_probs: HashMap<String, f32>,
        tokens: Vec<u32>,
        text: String,
        avg_logprob: f32,
        no_speech_prob: f32,
        temperature: f32,
        compression_ratio: f32,
    ) -> Self {
        Self {
            audio_features,
            language,
            language_probs,
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature,
            compression_ratio,
        }
    }
}


pub struct WhisperHelper<B: Backend> {
    // model
    pub config: WhisperConfig,
    pub model: Whisper<B>,
    pub tokenizer: Rc<Gpt2Tokenizer>,
    pub kind: WhichModel,
}

impl<B: Backend> WhisperHelper<B> {
    pub fn new(model_kind: WhichModel, device: &B::Device) -> WhisperHelper<B> {
        let hf_api = hf_hub::api::sync::ApiBuilder::new().build().unwrap();
        let (model_id, _) = model_kind.model_and_revision();
        let model_filename = hf_api
            .model(model_id.to_string())
            .get("pytorch_model.bin")
            .unwrap();
        let tokenizer_json_filename = hf_api.model(model_id.to_string()).get("tokenizer.json").unwrap();
        let config = model_kind.load_config().expect("can not load config.");
        let load_args = burn_import::pytorch::LoadArgs::new(model_filename)
            // .with_debug_print()
            // adapt to hugging face model. :-))
            .with_key_remap("model.(.*)", "$1")
            .with_key_remap(".layers.", ".blocks.")
            .with_key_remap(".encoder_attn.k_proj.", ".cross_attn.key.")
            .with_key_remap(".encoder_attn.out_proj.", ".cross_attn.out.")
            .with_key_remap(".encoder_attn.q_proj.", ".cross_attn.query.")
            .with_key_remap(".encoder_attn.v_proj.", ".cross_attn.value.")
            .with_key_remap(".encoder_attn_layer_norm.", ".cross_attn_ln.")
            .with_key_remap(
                r"encoder\.blocks\.(.*)\.self_attn\.out_proj\.",
                r"encoder.blocks.$1.attn.out.",
            )
            .with_key_remap(
                r"encoder\.blocks\.(.*)\.self_attn\.q_proj\.",
                r"encoder.blocks.$1.attn.query.",
            )
            .with_key_remap(
                r"encoder\.blocks\.(.*)\.self_attn\.v_proj\.",
                r"encoder.blocks.$1.attn.value.",
            )
            .with_key_remap(
                r"encoder\.blocks\.(.*)\.self_attn\.k_proj\.",
                r"encoder.blocks.$1.attn.key.",
            )
            .with_key_remap(
                r"encoder\.blocks\.(.*)\.self_attn_layer_norm\.",
                r"encoder.blocks.$1.attn_ln.",
            )
            .with_key_remap(
                r"decoder\.blocks\.(.*)\.self_attn\.out_proj\.",
                r"decoder.blocks.$1.attn.out.",
            )
            .with_key_remap(
                r"decoder\.blocks\.(.*)\.self_attn\.q_proj\.",
                r"decoder.blocks.$1.attn.query.",
            )
            .with_key_remap(
                r"decoder\.blocks\.(.*)\.self_attn\.v_proj\.",
                r"decoder.blocks.$1.attn.value.",
            )
            .with_key_remap(
                r"decoder\.blocks\.(.*)\.self_attn\.k_proj\.",
                r"decoder.blocks.$1.attn.key.",
            )
            .with_key_remap(
                r"decoder\.blocks\.(.*)\.self_attn_layer_norm\.",
                r"decoder.blocks.$1.attn_ln.",
            )
            .with_key_remap(".final_layer_norm.", ".mlp_ln.")
            .with_key_remap(".fc1.", ".mlp.0.")
            .with_key_remap(".fc2.", ".mlp.2.")
            .with_key_remap("encoder.layer_norm", "encoder.ln_post")
            .with_key_remap(".embed_positions.weight", ".positional_embedding")
            .with_key_remap("decoder.layer_norm.", "decoder.ln.")
            .with_key_remap("decoder.embed_tokens", "decoder.token_embedding")
            // adapt to burn's sequential model
            .with_key_remap("mlp.0", "mlp0")
            .with_key_remap("mlp.2", "mlp2");
        // .with_top_level_key("model_state_dict");
        let model =
            burn_import::pytorch::PyTorchFileRecorder::<burn::record::HalfPrecisionSettings>::new()
                .load(load_args, device)
                .map(|record| burn::module::Module::load_record(config.init(device), record))
                .expect("msg");

        let tokenizer = Rc::new(Gpt2Tokenizer::new(tokenizer_json_filename).expect("msg"));

        Self {
            config,
            model,
            tokenizer,
            kind: model_kind,
        }
    }

    /// Returns the token id for the selected language.

    /// Detect the spoken language in the audio, and return them as list of strings, along with the ids
    /// of the most probable language tokens and the probability distribution over all language tokens.
    /// This is performed outside the main decode loop in order to not interfere with kv-caching.
    pub fn detect_language(&self, mel: &Tensor<B, 3>) -> (Vec<Language>, Vec<f32>) {
        let [n_audio, _, _seq_len] = mel.dims();
        if !self.kind.is_multilingual() {
            return (vec![Language::English; n_audio], vec![1.0f32; n_audio]);
        }

        // mel = model.encoder(mel)
        let mel = self.model.forward_encoder(mel.clone());
        let device = &mel.device();
        let language_token_ids = crate::token::LANGUAGES
            .iter()
            .map(|t| {
                self.tokenizer
                    .special_token(SpecialToken::Language(Language::from_str(t).unwrap()))
                    .unwrap() as usize
            })
            .collect::<Vec<_>>();

        let sot_token = self
            .tokenizer
            .special_token(SpecialToken::StartofTranscript)
            .unwrap();
        // n_audio = mel.shape[0]
        // let n_audio = mel.dims()[0];
        // x = torch.tensor([[tokenizer.sot]] * n_audio).to(mel.device)  # [n_audio, 1]
        let x = Tensor::<B, 2, burn::tensor::Int>::full([n_audio, 1], sot_token as i32, device);
        let logits = self.model.forward_decoder(x, mel);
        let logits_dims = logits.dims();
        let logits = logits.slice([0..logits_dims[0], 0..1]).reshape([logits_dims[0], logits_dims[2]]);
        let logits_dims = logits.dims();

        let logits_dims_last = *logits_dims.last().unwrap();
        // # collect detected languages; suppress all non-language tokens
        // mask = torch.ones(logits.shape[-1], dtype=torch.bool)
        let mut mask_base = vec![true; logits_dims_last];

        // mask[list(tokenizer.all_language_tokens)] = False
        for l_token_id in language_token_ids {
            mask_base[l_token_id] = false;
        }
        let mask = Tensor::<B, 2, Bool>::from_data(
            burn::tensor::Data::new(mask_base, [1, logits_dims_last].into()),
            &device,
        );

        // logits[:, mask] = -np.inf
        let mask = mask
            .repeat(0, logits_dims[0])
            .reshape(logits_dims);
        let logits = logits.mask_fill(mask.clone(), -f32::INFINITY);
        // language_tokens = logits.argmax(dim=-1)
        let _language_tokens = logits.clone().argmax(1);
        // language_token_probs = logits.softmax(dim=-1).cpu()
        let language_token_probs = burn::tensor::activation::softmax(logits, 1); // dim= -1
        let (props, indices) = language_token_probs.clone().max_dim_with_indices(1);

        let props = props.into_data();
        let indices = indices.into_data();
        let mut res_language = vec![];
        let mut res_props = vec![];
        for (i, j) in indices.value.into_iter().zip(props.value.into_iter()) {
            let lang_token = self.tokenizer.id_to_token(i.to_u32().unwrap()).unwrap();
            let token_len = lang_token.len();
            res_language.push(Language::from_str(&lang_token[2..token_len - 2]).unwrap());
            res_props.push(j.to_f32().unwrap());
        }
        return (res_language, res_props);
    }


    fn init_decoder(&self, decoding_options: &DecodingOptions) -> Box<dyn TokenDecoder<B>> {
        let eot = self.tokenizer.special_token(SpecialToken::EndofText).unwrap();
        let decoder: GreedyDecoder = GreedyDecoder::new(decoding_options.temperature, eot as i32);
        return Box::new(decoder);
    }


    /// return the initial tokens
    /// ## Args
    /// - `audio_languages`: [n_audio]
    /// ## Return
    /// - `tokens`: n_batch * 3 tokens in a Vec<u32>
    fn get_initial_tokens(&self, audio_languages: Vec<Language>) -> Vec<u32> {
        let transcription_token = self.tokenizer.special_token(SpecialToken::Transcribe).unwrap();
        let sot_token = self.tokenizer.special_token(SpecialToken::StartofTranscript).unwrap();
        let mut res = vec![];
        for language in audio_languages {
            let lang_token = self.tokenizer.special_token(SpecialToken::Language(language)).unwrap();
            res.extend(
                vec![sot_token, lang_token, transcription_token]
            );
        };
        return res;
    }

    fn get_suppress_token(&self, decoding_options: &DecodingOptions) -> Vec<u32> {
        let mut suppress_token: Vec<u32> = vec![];
        if let Some(user_suppress_token) = &decoding_options.suppress_tokens {
            let user_tokens = match user_suppress_token {
                UserSuppressTokens::Text(s) => {
                    s.split(',').map(|x| {
                        x.parse::<i32>().unwrap()
                    }).collect::<Vec<_>>()
                }
                UserSuppressTokens::Tokens(s) => {
                    s.clone().into_iter().collect()
                }
            };
            if user_tokens.contains(&-1) {
                suppress_token = user_tokens.into_iter().filter(|x| *x >= 0).map(|x| x as u32).collect();
                suppress_token.extend(self.tokenizer.non_speech_tokens());
            }
        };
        let special_tokens = [
            SpecialToken::Transcribe,
            SpecialToken::Translate,
            SpecialToken::StartofTranscript,
            SpecialToken::StartofPrev,
            SpecialToken::StartofLM,
            SpecialToken::NoSpeech
        ];
        for token in special_tokens {
            if let Some(t) = self.tokenizer.special_token(token) {
                suppress_token.push(t);
            }
        }
        suppress_token.sort();
        suppress_token
    }


    pub fn run_mels(&self, mels: Tensor<B, 3>, decoding_options: DecodingOptions) -> Vec<DecodingResult<B>> {
        // ====== decoder args ======
        let n_group = decoding_options.beam_size.unwrap_or(
            decoding_options.best_of.unwrap_or(1)
        );

        let [n_batch, n_mel, _n_ctx] = mels.dims();
        let n_ctx = self.model.decoder_ctx_size();
        let sample_len = decoding_options.sample_len.unwrap_or(n_ctx / 2);
        // let sot_sequence

        let langs_tokens = if let Some(language) = &decoding_options.language {
            let lang = Language::from_str(language).unwrap_or(Language::English);
            vec![lang; n_batch]
        } else {
            let (langs, _) = self.detect_language(&mels);
            langs
        };
        let initial_tokens = self.get_initial_tokens(langs_tokens.clone());

        let sample_begin = initial_tokens.len() / n_batch;
        let sot_token = self.tokenizer.special_token(SpecialToken::StartofTranscript).unwrap();
        let _sot_index = initial_tokens.iter().find(|&&x| x == sot_token).unwrap();

        // let inference
        let sequence_ranker = TakeFirstGroup::new();
        // todo: beam search
        let decoder = self.init_decoder(&decoding_options);

        // logit filters: applies various rules to suppress or penalize certain tokens
        let mut logit_filters: Vec<Box<dyn LogitFilter<B>>> = vec![];
        if decoding_options.suppress_blank {
            logit_filters.push(Box::new(SuppressBlank::new(self.tokenizer.clone(), sample_begin)));
        }
        if decoding_options.suppress_tokens.is_some() {
            logit_filters.push(Box::new(crate::decoding::logit_filter::SuppressTokens::new(self.get_suppress_token(&decoding_options))));
        }


        // ======= decoder args end =
        println!("mel info: n_batch = {n_batch}, n_mel = {n_mel}, n_ctx = {n_ctx}");

        let device = mels.device();
        let audio_features = self.model.forward_encoder(mels);

        let n_ctx_max_encoder = self.model.encoder_ctx_size();
        let _n_ctx_max_decoder = self.model.decoder_ctx_size();

        let padding = 0;
        if n_ctx + padding > n_ctx_max_encoder {
            println!(
                "Audio has length of {} which exceeds maximum length {}. It will be clipped.",
                n_ctx + padding,
                n_ctx_max_encoder
            );
        }

        let end_token = self.tokenizer.special_token(SpecialToken::EndofText).unwrap();
        let sot_token = self.tokenizer.special_token(SpecialToken::StartofTranscript).unwrap();
        let no_speech_token = self.tokenizer.special_token(SpecialToken::NoSpeech);

        let initial_tokens = initial_tokens.iter().map(|&x| x as i32).collect::<Vec<_>>();
        let mut tokens: Tensor<B, 2, Int> = Tensor::from_ints(&*initial_tokens, &device)
            .reshape([n_batch, sample_begin])
            //  repeat text tensors by the group size, for beam search or best-of-n sampling
            //  tokens = tokens.repeat_interleave(self.n_group, dim=0).to(audio_features.device)
            .unsqueeze_dim::<3>(1)
            .repeat(1, n_group)
            .reshape([n_batch * n_group, 3]);

        // println!("initial_tokens = {tokens}");


        let mut sum_logprobs = Tensor::<B, 2>::zeros([n_batch, 1], &device);
        let mut no_speech_probs = Tensor::<B, 1>::full([n_batch], f32::INFINITY, &device);
        for i in 0..sample_len {
            let logits = self.model.forward_decoder(tokens.clone(), audio_features.clone());
            if i == 0 && no_speech_token.is_some() {
                let probs_at_sot =
                    softmax(logits.clone().slice([0..n_batch, (sot_token as usize)..((sot_token as usize) + 1)]), 2);
                no_speech_probs = probs_at_sot.slice(
                    [0..n_batch, 0..1, (no_speech_token.unwrap() as usize)..((no_speech_token.unwrap() as usize) + 1)]
                ).reshape([n_batch]);
            }
            let [n_audio, n_group, vol_size] = logits.dims();
            let mut logits = logits.slice([0..n_batch, (n_group - 1)..n_group]).reshape([n_audio, vol_size]);

            for filter in &logit_filters {
                logits = filter.apply(logits, &tokens);
            }
            let (next_tokens, completed) = decoder.update(tokens, logits, &mut sum_logprobs);
            // println!("next_tokens = {next_tokens}, completed = {completed}");
            tokens = next_tokens;
            if completed {
                break;
            }
        }


        // # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        // audio_features = audio_features[:: self.n_group]
        // no_speech_probs = no_speech_probs[:: self.n_group]
        // assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        let [n_batch_mul_n_group, seq_len] = tokens.dims();
        assert_eq!(n_batch_mul_n_group, n_batch * n_group);
        let tokens = tokens.reshape([n_batch, n_group, seq_len]);
        let sum_logprobs = sum_logprobs.reshape([n_batch, n_group]);

        let (tokens, _sum_logprops) = decoder.finalize(tokens, sum_logprobs.clone());
        let [_, _, seq_len] = tokens.dims();

        let tokens_data = tokens.into_data().value.into_iter().map(|x| x.to_u32().unwrap()).collect::<Vec<_>>();

        let mut tokens = vec![];
        for batch_idx in 0..n_batch {
            let mut group_tokens = vec![];
            for group_idx in 0..n_group {
                // let l=sample_begin;
                let mut r = seq_len;
                for r0 in sample_begin..seq_len {
                    if tokens_data[batch_idx * (n_group * seq_len) + group_idx * seq_len + r0] == end_token {
                        r = r0;
                        break;
                    }
                }
                group_tokens.push(tokens_data[(batch_idx * (n_group * seq_len) + group_idx * seq_len + sample_begin)..(batch_idx * (n_group * seq_len) + group_idx * seq_len + r)].to_vec());
            }
            tokens.push(group_tokens);
        }

        // select the top-ranked sample in each group
        let selected = sequence_ranker.rank(&tokens, &sum_logprobs);
        let tokens = selected
            .iter()
            .zip(tokens.iter())
            .map(|(i, t)| {
                t[*i].clone()
            }).collect::<Vec<_>>();
        let mut texts = vec![];
        for token in &tokens {
            texts.push(self.tokenizer.decode(token, true).expect(""));
        }

        let sum_logprobs = sum_logprobs
            .to_data().value.iter()
            .map(|&x| x.to_f32().unwrap())
            .collect::<Vec<_>>()
            .chunks(n_group).into_iter()// Vec<[f32;n_group]>
            .zip(selected.iter())
            .map(|(lp, i)| {
                lp[*i]
            }).collect::<Vec<_>>();

        let avg_logprobs = sum_logprobs.iter().zip(tokens.iter())
            .map(|(&lp, t)| {
                lp / ((t.len() + 1) as f32)
            }).collect::<Vec<_>>();


        let res = texts.into_iter()
            .zip(langs_tokens.into_iter())
            .zip(tokens.into_iter())
            .zip(audio_features.iter_dim(0))
            .zip(avg_logprobs.into_iter())
            .zip(no_speech_probs.iter_dim(0))
            .map(|(((((text, language), tokens), features), avg_logprob), no_speech_prob)| {
                let [_, group, length] = features.dims();
                let audio_features = features.reshape([group, length]);

                DecodingResult::new(
                    audio_features,
                    language,
                    Default::default(),
                    tokens,
                    text,
                    avg_logprob,
                    no_speech_prob.into_scalar().to_f32().unwrap(),
                    decoding_options.temperature,
                    0.0,
                )
            }).collect::<Vec<_>>();
        return res;
    }

    pub fn run(&self, raw_wave: &[f32], mut batch_size: usize, decoding_options: DecodingOptions, device: &Device<B>) -> Vec<DecodingResult<B>> {
        if batch_size == 0 {
            batch_size = 4;
        }
        let res = raw_wave.chunks(batch_size * N_SAMPLES).map(|data| {
            let audio = if data.len() == batch_size * N_SAMPLES {
                Tensor::<B, 2>::from_floats(
                    burn::tensor::Data::new(Vec::from(data), [batch_size, N_SAMPLES].into()),
                    device,
                )
            } else {
                let mut data = Vec::from(data);
                let pad_size = if data.len() % N_SAMPLES == 0 { 0 } else { N_SAMPLES - data.len() % N_SAMPLES };
                for _i in 0..pad_size {
                    data.push(0.0f32);
                }
                let data_len = data.len();
                Tensor::<B, 2>::from_floats(
                    burn::tensor::Data::new(data, [data_len / N_SAMPLES, N_SAMPLES].into()),
                    device,
                )
            };
            // we dont need pad because we have done.
            // let audio = pad_or_trim(&audio, N_SAMPLES);
            let mel = log_mel_spectrogram(audio);
            self.run_mels(mel, decoding_options.clone())
        }).flatten().collect();
        res
    }
    fn _decode_with_fallback(
        &self,
        segment: Tensor<B, 3>,
        temperatures: &Vec<f32>,
        decode_options: DecodingOptions,
    ) -> Vec<DecodingResult<B>> {
        let mut decode_result = vec![];

        for &t in temperatures.iter() {
            let mut options = decode_options.clone(); // Assuming DecodingOptions is Clone
            if t > 0.0 {
                options.beam_size = None;
                options.patience = None;
            } else {
                options.best_of = None;
            }
            options.temperature = t;
            decode_result = self.run_mels(segment.clone(), options);

            let needs_fallback = false;
            // if let Some(threshold) =  options.compression_ratio_threshold {
            //     if decode_result.compression_ratio > threshold {
            //         needs_fallback = true;
            //     }
            // }
            // if let Some(threshold) = logprob_threshold {
            //     if decode_result.avg_logprob < threshold {
            //         needs_fallback = true;
            //     }
            // }
            // if let Some(threshold) = no_speech_threshold {
            //     if decode_result.no_speech_prob > threshold {
            //         needs_fallback = false; // This looks like a potential bug in the original Python code, double-check logic
            //     }
            // }
            if !needs_fallback {
                return decode_result;
            }
        }

        // Fallback or final decode result handling
        // Assuming returning the last decode_result if all attempts fall back
        decode_result
    }
}

#[cfg(test)]
mod test {
    use burn::prelude::Tensor;
    use burn_wgpu::{AutoGraphicsApi, Wgpu, WgpuDevice};
    use tokio;

    use crate::decoding::DecodingOptions;

    use crate::whsiper_helper::{WhichModel, WhisperHelper};

    #[tokio::test]
    async fn test_detect_language() {
        cfg_if::cfg_if! {
            if #[cfg(feature = "wgpu-backend")] {
                type CurBackend = Wgpu<AutoGraphicsApi, f32, i32>;
                let device = WgpuDevice::BestAvailable;
            } else if #[cfg(feature = "torch-backend")] {
                type CurBackend = LibTorch<f32>;
                let device = LibTorchDevice::Cuda(0);
            }
        }

        let decode_options = DecodingOptions::default();
        let m: WhisperHelper<CurBackend> = WhisperHelper::new(WhichModel::Base, decode_options, &device);
    }


    #[test]
    fn test_repeat() {
        cfg_if::cfg_if! {
            if #[cfg(feature = "wgpu-backend")] {
                type CurBackend = Wgpu<AutoGraphicsApi, f32, i32>;
                let device = WgpuDevice::BestAvailable;
            } else if #[cfg(feature = "torch-backend")] {
                type CurBackend = LibTorch<f32>;
                let device = LibTorchDevice::Cuda(0);
            }
        }


        let input = Tensor::<CurBackend, 2>::from_data(
            [
                [10.0, 20.0, 30.0],
                [11.0, 22.0, 33.0],
            ],
            &device,
        );
        let input = input.unsqueeze_dim::<3>(1).repeat(1, 4).reshape([8, 3]);
        println!("input = {input}");
    }
}
