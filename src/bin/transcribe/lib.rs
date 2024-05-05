use std::{collections::HashMap, marker::PhantomData};

use burn::tensor::{backend::Backend, Device, Int, Tensor};
use num_traits::{clamp, Zero};
use whisper::{
    decoding::DecodingOptions,
    model::Whisper,
};
//
// // 需要先定义或引入以下类型
// #[derive(Debug)]
// struct Inference;
//
// #[derive(Debug)]
// struct SequenceRanker;
//
// #[derive(Debug)]
// struct TokenDecoder;
//
// #[derive(Debug)]
// struct LogitFilter;
//
// #[derive(Debug)]
// struct DecodingTask<B: Backend> {
//     inference: Inference,
//     sequence_ranker: SequenceRanker,
//     decoder: TokenDecoder,
//     logit_filters: Vec<LogitFilter>,
//     a: PhantomData<B>,
// }
//
// impl<B: Backend> DecodingTask<B> {
//     fn new(model: &Whisper<B>, options: &DecodingOptions) -> Self {
//         // 初始化逻辑
//         DecodingTask::<B> {
//             inference: Inference,            // 需要具体实现
//             sequence_ranker: SequenceRanker, // 需要具体实现
//             decoder: TokenDecoder,           // 需要具体实现
//             logit_filters: vec![],
//             a: PhantomData, // 需要根据实际情况填充
//         }
//     }
//
//     fn verify_options(&self, options: DecodingOptions) -> DecodingOptions {
//         // 验证选项的逻辑
//         options
//     }
//
//     fn get_initial_tokens(&self) -> Vec<i32> {
//         // 获取初始令牌的逻辑
//         vec![] // 示例
//     }
//
//     fn get_suppress_tokens(&self) -> Vec<i32> {
//         // 获取要抑制的令牌的逻辑
//         vec![] // 示例
//     }
//
//     fn get_audio_features(&self, mel: &Tensor<B, 2>) -> Tensor<B, 2> {
//         // 获取音频特征的逻辑
//         mel.clone() // 示例
//     }
//
//     fn detect_language(
//         &self,
//         audio_features: &Tensor<B, 2>,
//         tokens: &Tensor<B, 2>,
//     ) -> (Vec<String>, Vec<f32>) {
//         // 语言检测逻辑
//         (vec![], vec![]) // 示例
//     }
//
//     fn main_loop(
//         &self,
//         audio_features: &Tensor<B, 2>,
//         tokens: &Tensor<B, 2>,
//     ) -> (Vec<i32>, f32, f32) {
//         // 主循环的逻辑
//         (vec![], 0.0, 0.0) // 示例
//     }
//
//     pub fn run(&self, mel: &Tensor<B, 3>) -> DecodingResult<B> {
//         todo!()
//         // no_grad(|| {
//         //     // 运行解码任务的逻辑
//         //     vec![] // 示例
//         // })
//     }
// }

// Tokenizer dependent bits.
pub const SOT_TOKEN: &str = "<|startoftranscript|>";
pub const TRANSCRIBE_TOKEN: &str = "<|transcribe|>";
pub const TRANSLATE_TOKEN: &str = "<|translate|>";
pub const NO_TIMESTAMPS_TOKEN: &str = "<|notimestamps|>";
pub const EOT_TOKEN: &str = "<|endoftext|>";
pub const NO_SPEECH_TOKENS: [&str; 2] = ["<|nocaptions|>", "<|nospeech|>"];
//
// fn decode_with_fallback<B: Backend>(
//     segment: &Tensor<B, 3>,
//     model: &Whisper<B>,
//     temperatures: &Vec<f32>,
//     decode_options: &whisper::decoding::DecodingOptions,
//     compression_ratio_threshold: Option<f32>,
//     logprob_threshold: Option<f32>,
//     no_speech_threshold: Option<f32>,
//     device: &Device<B>,
// ) -> DecodingResult<B> {
//     let mut decode_result = DecodingResult::<B>::new(device);
//
//     for &t in temperatures.iter() {
//         let mut options = decode_options.clone(); // Assuming DecodingOptions is Clone
//         if t > 0.0 {
//             options.beam_size = None;
//             options.patience = None;
//         } else {
//             options.best_of = None;
//         }
//         options.temperature = t;
//         decode_result = DecodingTask::new(model, &options).run(segment);
//
//         let mut needs_fallback = false;
//         if let Some(threshold) = compression_ratio_threshold {
//             if decode_result.compression_ratio > threshold {
//                 needs_fallback = true;
//             }
//         }
//         if let Some(threshold) = logprob_threshold {
//             if decode_result.avg_logprob < threshold {
//                 needs_fallback = true;
//             }
//         }
//         if let Some(threshold) = no_speech_threshold {
//             if decode_result.no_speech_prob > threshold {
//                 needs_fallback = false; // This looks like a potential bug in the original Python code, double-check logic
//             }
//         }
//         if !needs_fallback {
//             return decode_result;
//         }
//     }
//
//     // Fallback or final decode result handling
//     // Assuming returning the last decode_result if all attempts fall back
//     decode_result
// }
//
// pub fn transcribe<B: Backend>(
//     whisper: &Whisper<B>,
//     _bpe: &whisper::token::Gpt2Tokenizer,
//     _lang: whisper::token::Language,
//     mut audio: Vec<f32>,
//     temperature: Vec<f32>, // (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
//
//     compression_ratio_threshold: Option<f32>, // 2.4
//     logprob_threshold: Option<f32>,           // -1
//     no_speech_threshold: Option<f32>,         // 0.6
//     // sample_rate: usize, 16k
//     clip_timestamps: Vec<(f32, f32)>,
//     decode_options: &whisper::decoding::DecodingOptions,
//     device: &Device<B>,
// ) {
//     // println!("begin transcribe");
//     // // Pad 30-seconds of silence to the input audio, for slicing
//     // for i in 0..N_SAMPLES {
//     //     audio.push(i as f32);
//     // }
//     // let input_len = audio.len();
//     // let audio = Tensor::<B, 2>::from_floats(
//     //     burn::tensor::Data::new(audio, [1, input_len].into()),
//     //     &device,
//     // )
//     // .to_device(&device);
//     //
//     // let mel = log_mel_spectrogram::<B>(audio);
//     // let (a, b) = detect_language(&whisper, &_bpe, &mel);
//     // println!("detect: {a:?}, props: {b}");
//     // return;
//     // return;
//     // let seek_clips = clip_timestamps
//     //     .iter()
//     //     .map(|(s, t)| {
//     //         return (
//     //             (s.round() as usize) * FRAMES_PER_SECOND,
//     //             (t.round() as usize) * FRAMES_PER_SECOND,
//     //         );
//     //     })
//     //     .collect::<Vec<_>>();
//     //
//     // if seek_clips.len() == 0 {
//     //     return;
//     // }
//     //
//     // let content_frames = *(mel.shape().dims.last().unwrap());
//     // let content_duration = ((content_frames * HOP_LENGTH) as f32) / (SAMPLE_RATE as f32);
//     //
//     // let punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、";
//     //
//     // let mut seek = seek_clips[0].0;
//     // // let mut all_tokens = vec![];
//     //
//     // for (seek_clip_start, seek_clip_end) in seek_clips {
//     //     seek = clamp(seek, seek_clip_start, seek_clip_start);
//     //
//     //     let time_offset = ((seek * HOP_LENGTH) as f32) / (SAMPLE_RATE as f32);
//     //     let window_end_time = (((seek + N_FRAMES) * HOP_LENGTH) as f32) / (SAMPLE_RATE as f32);
//     //     let segment_size = std::cmp::min(
//     //         N_FRAMES,
//     //         std::cmp::min(content_frames - seek, seek_clip_end - seek),
//     //     );
//     //     let mel_dim_0 = mel.shape().dims[1];
//     //     let mel_segment = mel
//     //         .clone()
//     //         .slice([0..1, 0..mel_dim_0, seek..seek + segment_size]);
//     //     let segment_duration = (segment_size * HOP_LENGTH) as f32 / (SAMPLE_RATE as f32);
//     //     let padded_mel_segment = whisper::audio::pad_or_trim(&mel_segment, N_FRAMES);
//     //     let res = decode_with_fallback(
//     //         &padded_mel_segment,
//     //         &whisper,
//     //         &temperature,
//     //         decode_options,
//     //         compression_ratio_threshold,
//     //         logprob_threshold,
//     //         no_speech_threshold,
//     //         device,
//     //     );
//     //     println!("{:?}", res);
//     // }
// }
//
//
// #[cfg(test)]
// mod test{
//     #[test]
//     fn test_detect_language(){
//
//     }
// }