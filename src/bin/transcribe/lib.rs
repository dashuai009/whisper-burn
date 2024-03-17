use std::{collections::HashMap, marker::PhantomData};

use burn::tensor::{backend::Backend, Bool, Data, Device, Int, Tensor};
use burn::tensor::activation::softmax;
use num_traits::{clamp, Zero};
use whisper::{
    audio::{log_mel_spectrogram, FRAMES_PER_SECOND, HOP_LENGTH, N_FRAMES, N_SAMPLES, SAMPLE_RATE},
    decoding::DecodingOptions,
    model::Whisper,
};
use whisper::token::{Gpt2Tokenizer, Language, SpecialToken};


#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WhichModel {
    Tiny,
    // #[value(name = "tiny.en")]
    TinyEn,
    Base,
    // #[value(name = "base.en")]
    BaseEn,
    Small,
    // #[value(name = "small.en")]
    SmallEn,
    Medium,
    // #[value(name = "medium.en")]
    MediumEn,
    Large,
    LargeV2,
    LargeV3,
    // #[value(name = "distil-medium.en")]
    DistilMediumEn,
    // #[value(name = "distil-large-v2")]
    DistilLargeV2,
}

impl WhichModel {
    fn is_multilingual(&self) -> bool {
        match self {
            Self::Tiny
            | Self::Base
            | Self::Small
            | Self::Medium
            | Self::Large
            | Self::LargeV2
            | Self::LargeV3
            | Self::DistilLargeV2 => true,
            Self::TinyEn | Self::BaseEn | Self::SmallEn | Self::MediumEn | Self::DistilMediumEn => {
                false
            }
        }
    }

    fn model_and_revision(&self) -> (&'static str, &'static str) {
        match self {
            Self::Tiny => ("openai/whisper-tiny", "main"),
            Self::TinyEn => ("openai/whisper-tiny.en", "refs/pr/15"),
            Self::Base => ("openai/whisper-base", "refs/pr/22"),
            Self::BaseEn => ("openai/whisper-base.en", "refs/pr/13"),
            Self::Small => ("openai/whisper-small", "main"),
            Self::SmallEn => ("openai/whisper-small.en", "refs/pr/10"),
            Self::Medium => ("openai/whisper-medium", "main"),
            Self::MediumEn => ("openai/whisper-medium.en", "main"),
            Self::Large => ("openai/whisper-large", "refs/pr/36"),
            Self::LargeV2 => ("openai/whisper-large-v2", "refs/pr/57"),
            Self::LargeV3 => ("openai/whisper-large-v3", "main"),
            Self::DistilMediumEn => ("distil-whisper/distil-medium.en", "main"),
            Self::DistilLargeV2 => ("distil-whisper/distil-large-v2", "main"),
        }
    }
}

#[derive(Debug)]
struct DecodingResult<B: Backend> {
    audio_features: Tensor<B, 2>,
    language: String,
    language_probs: HashMap<String, f32>,
    //  Optional[Dict[str, float]] = None
    tokens: Vec<i32>,
    text: String,
    avg_logprob: f32,
    no_speech_prob: f32,
    temperature: f32,
    compression_ratio: f32,
}

impl<B: Backend> DecodingResult<B> {
    fn new(device: &Device<B>) -> Self {
        Self {
            audio_features: Tensor::<B, 2>::zeros([2, 3], device),
            language: Default::default(),
            language_probs: Default::default(),
            tokens: Default::default(),
            text: Default::default(),
            avg_logprob: Default::default(),
            no_speech_prob: Default::default(),
            temperature: Default::default(),
            compression_ratio: Default::default(),
        }
    }
}

// 需要先定义或引入以下类型
#[derive(Debug)]
struct Inference;

#[derive(Debug)]
struct SequenceRanker;

#[derive(Debug)]
struct TokenDecoder;

#[derive(Debug)]
struct LogitFilter;

#[derive(Debug)]
struct DecodingTask<B: Backend> {
    inference: Inference,
    sequence_ranker: SequenceRanker,
    decoder: TokenDecoder,
    logit_filters: Vec<LogitFilter>,
    a: PhantomData<B>,
}

impl<B: Backend> DecodingTask<B> {
    fn new(model: &Whisper<B>, options: &DecodingOptions) -> Self {
        // 初始化逻辑
        DecodingTask::<B> {
            inference: Inference,            // 需要具体实现
            sequence_ranker: SequenceRanker, // 需要具体实现
            decoder: TokenDecoder,           // 需要具体实现
            logit_filters: vec![],
            a: PhantomData, // 需要根据实际情况填充
        }
    }

    fn verify_options(&self, options: DecodingOptions) -> DecodingOptions {
        // 验证选项的逻辑
        options
    }

    fn get_initial_tokens(&self) -> Vec<i32> {
        // 获取初始令牌的逻辑
        vec![] // 示例
    }

    fn get_suppress_tokens(&self) -> Vec<i32> {
        // 获取要抑制的令牌的逻辑
        vec![] // 示例
    }

    fn get_audio_features(&self, mel: &Tensor<B, 2>) -> Tensor<B, 2> {
        // 获取音频特征的逻辑
        mel.clone() // 示例
    }

    fn detect_language(
        &self,
        audio_features: &Tensor<B, 2>,
        tokens: &Tensor<B, 2>,
    ) -> (Vec<String>, Vec<f32>) {
        // 语言检测逻辑
        (vec![], vec![]) // 示例
    }

    fn main_loop(
        &self,
        audio_features: &Tensor<B, 2>,
        tokens: &Tensor<B, 2>,
    ) -> (Vec<i32>, f32, f32) {
        // 主循环的逻辑
        (vec![], 0.0, 0.0) // 示例
    }

    pub fn run(&self, mel: &Tensor<B, 3>) -> DecodingResult<B> {
        todo!()
        // no_grad(|| {
        //     // 运行解码任务的逻辑
        //     vec![] // 示例
        // })
    }
}


// Tokenizer dependent bits.
pub const SOT_TOKEN: &str = "<|startoftranscript|>";
pub const TRANSCRIBE_TOKEN: &str = "<|transcribe|>";
pub const TRANSLATE_TOKEN: &str = "<|translate|>";
pub const NO_TIMESTAMPS_TOKEN: &str = "<|notimestamps|>";
pub const EOT_TOKEN: &str = "<|endoftext|>";
pub const NO_SPEECH_TOKENS: [&str; 2] = ["<|nocaptions|>", "<|nospeech|>"];


/// Returns the token id for the selected language.
pub fn detect_language<B: Backend>(
    model: &mut Whisper<B>,
    tokenizer: &Gpt2Tokenizer,
    mel: &Tensor<B, 3>,
) -> (Language, f32) {
    let [_bsize, _, seq_len] = mel.dims();
    let mel = model.forward_encoder(mel.clone());
    // let mel = mel.clone().narrow(
    //     2,
    //     0,
    //     usize::min(seq_len, model.config().max_source_positions),
    // )?;
    let device = &mel.device();
    let language_token_ids = whisper::token::LANGUAGES
        .iter()
        .map(|t| tokenizer.special_token(SpecialToken::Language(Language::from_str(t).unwrap())).unwrap())
        .collect::<Vec<_>>();
    let sot_token = tokenizer.special_token(SpecialToken::StartofTranscript).unwrap();
    let n_audio = mel.dims()[0];
    let x = Tensor::<B, 2, Int>::full([n_audio, 1], sot_token as i32, device,
    );
    let logits = model.forward_decoder(x, mel);
    let logits_dims = logits.dims();
    let mask = Tensor::<B, 1>::ones([logits_dims[2]], device);
    for l_token_id in language_token_ids {
        mask.to_data().value[l_token_id] = B::FloatElem::zero();
    }

    let language_tokens = logits.clone().argmax(2);// language_tokens = logits.argmax(dim=-1)
    let language_token_probs = softmax(logits, 2);// dim= -1

    let (a, b) =  language_token_probs.max_dim_with_indices(2);
    return (Language::English, 1.0f32);


    // logits[:, mask] = -np.inf
    //
    // language_token_probs = logits.softmax(dim=-1).cpu()
    // language_probs = [
    //     {
    //         c: language_token_probs[i, j].item()
    //         for j, c in zip(tokenizer.all_language_tokens, tokenizer.all_language_codes)
    //     }
    // for i in range(n_audio)
    // ]
    //
    // if single:
    //     language_tokens = language_tokens[0]
    // language_probs = language_probs[0]
    //
    // return language_tokens, language_probs

    // let language_token_ids = Tensor::new(language_token_ids.as_slice(), device)?;
    // let ys = model.decoder_forward(&tokens, &audio_features, true)?;
    // let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
    // let logits = logits.index_select(&language_token_ids, 0)?;
    // let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;
    // let probs = probs.to_vec1::<f32>()?;
    // let mut probs = whisper::token::LANGUAGES.iter().zip(probs.iter()).collect::<Vec<_>>();
    // probs.sort_by(|(_, p1), (_, p2)| p2.total_cmp(p1));
    // for ((_, language), p) in probs.iter().take(5) {
    //     println!("{language}: {p}")
    // }
    // let language = crate::token_id(tokenizer, &format!("<|{}|>", probs[0].0.0))?;
    // Ok(language)
}

fn decode_with_fallback<B: Backend>(
    segment: &Tensor<B, 3>,
    model: &Whisper<B>,
    temperatures: &Vec<f32>,
    decode_options: &whisper::decoding::DecodingOptions,
    compression_ratio_threshold: Option<f32>,
    logprob_threshold: Option<f32>,
    no_speech_threshold: Option<f32>,
    device: &Device<B>,
) -> DecodingResult<B> {
    let mut decode_result = DecodingResult::<B>::new(device);

    for &t in temperatures.iter() {
        let mut options = decode_options.clone(); // Assuming DecodingOptions is Clone
        if t > 0.0 {
            options.beam_size = None;
            options.patience = None;
        } else {
            options.best_of = None;
        }
        options.temperature = t;
        decode_result = DecodingTask::new(model, &options).run(segment);

        let mut needs_fallback = false;
        if let Some(threshold) = compression_ratio_threshold {
            if decode_result.compression_ratio > threshold {
                needs_fallback = true;
            }
        }
        if let Some(threshold) = logprob_threshold {
            if decode_result.avg_logprob < threshold {
                needs_fallback = true;
            }
        }
        if let Some(threshold) = no_speech_threshold {
            if decode_result.no_speech_prob > threshold {
                needs_fallback = false; // This looks like a potential bug in the original Python code, double-check logic
            }
        }
        if !needs_fallback {
            return decode_result;
        }
    }

    // Fallback or final decode result handling
    // Assuming returning the last decode_result if all attempts fall back
    decode_result
}

pub fn transcribe<B: Backend>(
    whisper: &Whisper<B>,
    _bpe: &whisper::token::Gpt2Tokenizer,
    _lang: whisper::token::Language,
    mut audio: Vec<f32>,
    temperature: Vec<f32>, // (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    compression_ratio_threshold: Option<f32>, // 2.4
    logprob_threshold: Option<f32>, // -1
    no_speech_threshold: Option<f32>, // 0.6
    // sample_rate: usize, 16k
    clip_timestamps: Vec<(f32, f32)>,
    decode_options: &whisper::decoding::DecodingOptions,
    device: &Device<B>,
) {
    // Pad 30-seconds of silence to the input audio, for slicing
    for i in 0..N_SAMPLES {
        audio.push(i as f32);
    }
    let input_len = audio.len();
    let audio = Tensor::<B, 2>::from_floats(
        burn::tensor::Data::new(audio, [1, input_len].into()),
        &device,
    )
        .to_device(&device);

    let mel = log_mel_spectrogram::<B>(audio);
    let seek_clips = clip_timestamps
        .iter()
        .map(|(s, t)| {
            return (
                (s.round() as usize) * FRAMES_PER_SECOND,
                (t.round() as usize) * FRAMES_PER_SECOND,
            );
        })
        .collect::<Vec<_>>();

    if seek_clips.len() == 0 {
        return;
    }

    let content_frames = *(mel.shape().dims.last().unwrap());
    let content_duration = ((content_frames * HOP_LENGTH) as f32) / (SAMPLE_RATE as f32);

    let punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、";

    let mut seek = seek_clips[0].0;
    // let mut all_tokens = vec![];

    for (seek_clip_start, seek_clip_end) in seek_clips {
        seek = clamp(seek, seek_clip_start, seek_clip_start);

        let time_offset = ((seek * HOP_LENGTH) as f32) / (SAMPLE_RATE as f32);
        let window_end_time = (((seek + N_FRAMES) * HOP_LENGTH) as f32) / (SAMPLE_RATE as f32);
        let segment_size = std::cmp::min(
            N_FRAMES,
            std::cmp::min(content_frames - seek, seek_clip_end - seek),
        );
        let mel_dim_0 = mel.shape().dims[1];
        let mel_segment = mel.clone().slice([0..1, 0..mel_dim_0, seek..seek + segment_size]);
        let segment_duration = (segment_size * HOP_LENGTH) as f32 / (SAMPLE_RATE as f32);
        let padded_mel_segment = whisper::audio::pad_or_trim(&mel_segment, N_FRAMES);
        let res = decode_with_fallback(
            &padded_mel_segment,
            &whisper,
            &temperature,
            decode_options,
            compression_ratio_threshold,
            logprob_threshold,
            no_speech_threshold,
            device,
        );
        println!("{:?}", res);
    }
}
