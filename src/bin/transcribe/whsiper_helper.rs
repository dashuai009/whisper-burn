use std::collections::HashMap;
use burn::tensor::backend::Backend;

use burn::tensor::{Bool, Int, Tensor};
use num_traits::{ToPrimitive, Zero};
use reqwest::{self, IntoUrl};

use std::fs::File;

use std::io::{copy, Write};

use std::path::{Path, PathBuf};
use std::rc::Rc;
use burn::prelude::{Data, Device};

use whisper::model::{Whisper, WhisperConfig};

use whisper::token::{Gpt2Tokenizer, Language, SpecialToken};

use burn::record::Recorder;
use whisper::decoding::{DecodingOptions, GreedyDecoder, TokenDecoder};
use whisper::decoding::logit_filter::{LogitFilter, SuppressBlank};

async fn download_from_url_to_file<
    T: IntoUrl + std::fmt::Debug,
    P: AsRef<Path> + std::fmt::Debug,
>(
    url: T,
    load_file_path: P,
) -> Result<(), reqwest::Error> {
    println!("download from [{url:?}] \n\t to {load_file_path:?}");
    let response = reqwest::get(url).await?;
    // 分离文件路径和文件名
    let perfix = load_file_path.as_ref().parent().unwrap();
    std::fs::create_dir_all(perfix).expect("msg");
    if response.status().is_success() {
        let mut dest = File::create(load_file_path).expect("msg");
        let content = response.bytes().await?;
        copy(&mut content.as_ref(), &mut dest).expect("");
    } else {
        println!("download error: {}", response.status());
    }
    return Ok(());
}

/// 模型种类。
/// 目前一共11种模型+2种distil出品的加速推理的模型
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhichModel {
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

    async fn download_to_dir<T: AsRef<Path>>(&self, dir: T) -> Result<(), reqwest::Error> {
        // 目标本地路径
        let local_path = dir.as_ref().join(self.model_local_path());
        if !local_path.exists() {
            let url = match self {
                Self::TinyEn => "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
                Self::Tiny => r"https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
                Self::BaseEn => r"https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
                Self::Base => r"https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
                Self::SmallEn => r"https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
                Self::Small => r"https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
                Self::MediumEn => r"https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
                Self::Medium => r"https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
                Self::LargeV1 => r"https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
                Self::LargeV2 => r"https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
                Self::LargeV3 => r"https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
                Self::DistilLargeV2 | Self::DistilMediumEn => todo!()
            };
            download_from_url_to_file(url, local_path).await?;
        }

        let tokenizer_json = dir.as_ref().join(self.tokenizer_json_path());
        if !tokenizer_json.exists() {
            let url = format!(
                "https://huggingface.co/{}/resolve/main/tokenizer.json?download=true",
                self.model_and_revision().0
            );
            download_from_url_to_file(url, tokenizer_json).await?;
        }
        Ok(())
    }

    fn model_local_path(&self) -> PathBuf {
        format!("{}/model.pt", self.as_str()).into()
    }

    fn config_path(&self) -> PathBuf {
        format!("{}/{}.cfg", self.as_str(), self.as_str()).into()
    }
    fn tokenizer_json_path(&self) -> PathBuf {
        format!("{}/tokenizer.json", self.as_str()).into()
    }

    fn as_str(&self) -> &'static str {
        match self {
            WhichModel::Tiny => "tiny",
            WhichModel::TinyEn => "tiny_en",
            WhichModel::Base => "base",
            WhichModel::BaseEn => "base_en",
            WhichModel::Small => "small",
            WhichModel::SmallEn => "small_en",
            WhichModel::Medium => "medium",
            WhichModel::MediumEn => "medium_en",
            WhichModel::LargeV1 => "large_v1",
            WhichModel::LargeV2 => "large_v2",
            WhichModel::LargeV3 => "large_v3",
            WhichModel::DistilMediumEn => todo!(),
            WhichModel::DistilLargeV2 => todo!(),
        }
    }

    fn is_downlaoded(&self) -> bool {
        return false;
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
            Self::LargeV1 => ("openai/whisper-large", "refs/pr/36"),
            Self::LargeV2 => ("openai/whisper-large-v2", "refs/pr/57"),
            Self::LargeV3 => ("openai/whisper-large-v3", "main"),
            Self::DistilMediumEn => ("distil-whisper/distil-medium.en", "main"),
            Self::DistilLargeV2 => ("distil-whisper/distil-large-v2", "main"),
        }
    }
}


#[derive(Clone)]
struct BeamSearchToken {
    token: usize,
    log_prob: f64,
}


#[derive(Debug)]
pub struct DecodingResult<B: Backend> {
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


pub struct WhisperHelper<B: Backend> {
    // model
    pub config: WhisperConfig,
    pub model: Whisper<B>,
    pub tokenizer: Rc<Gpt2Tokenizer>,
    pub kind: WhichModel,
    // decode
    pub decode_options: DecodingOptions,
}

impl<B: Backend> WhisperHelper<B> {
    pub async fn new(model_kind: WhichModel, decode_options: DecodingOptions, device: &B::Device) -> WhisperHelper<B> {
        let model_dir = Path::new("./model");
        model_kind.download_to_dir(model_dir).await.expect("msg");
        let config_path = model_dir.join(model_kind.config_path());
        let config = <WhisperConfig as burn::config::Config>::load(config_path).expect("msg");
        let load_args = burn_import::pytorch::LoadArgs::new(
            model_dir.join(model_kind.model_local_path()).into(),
        )
            // .with_debug_print()
            .with_key_remap("mlp.0", "mlp0")
            .with_key_remap("mlp.2", "mlp2")
            .with_top_level_key("model_state_dict");
        let model =
            burn_import::pytorch::PyTorchFileRecorder::<burn::record::HalfPrecisionSettings>::new()
                .load(load_args, device)
                .map(|record| burn::module::Module::load_record(config.init(device), record))
                .expect("msg");

        let tokenizer = Rc::new(
            Gpt2Tokenizer::new(model_dir.join(model_kind.tokenizer_json_path())).expect("msg")
        );

        Self {
            config,
            model,
            tokenizer,
            kind: model_kind,
            decode_options,
        }
    }

    /// Returns the token id for the selected language.

    /// Detect the spoken language in the audio, and return them as list of strings, along with the ids
    /// of the most probable language tokens and the probability distribution over all language tokens.
    /// This is performed outside the main decode loop in order to not interfere with kv-caching.
    pub fn detect_language(&self, mel: &Tensor<B, 3>) -> Vec<(Language, f32)> {
        if !self.kind.is_multilingual() {
            return vec![(Language::English, 1.0f32)];
        }
        // let mel = whisper::audio::pad_or_trim(mel, whisper::audio::N_FRAMES);
        let [n_audio, _, _seq_len] = mel.dims();

        // mel = model.encoder(mel)
        let mel = self.model.forward_encoder(mel.clone());
        let device = &mel.device();
        let language_token_ids = whisper::token::LANGUAGES
            .iter()
            .map(|t| {
                self.tokenizer
                    .special_token(SpecialToken::Language(Language::from_str(t).unwrap()))
                    .unwrap()
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
        let language_tokens = logits.clone().argmax(1);
        // language_token_probs = logits.softmax(dim=-1).cpu()
        let language_token_probs = burn::tensor::activation::softmax(logits, 1); // dim= -1
        let (props, indices) = language_token_probs.clone().max_dim_with_indices(1);

        let props = props.into_data();
        let indices = indices.into_data();
        let mut res = vec![];
        for (i, j) in indices.value.into_iter().zip(props.value.into_iter()) {
            let lang_token = self.tokenizer.id_to_token(i.to_u32().unwrap()).unwrap();
            let token_len = lang_token.len();
            res.push((Language::from_str(&lang_token[2..token_len - 2]).unwrap(), j.to_f32().unwrap()));
        }
        return res;
    }


    fn init_decoder(&self) -> Box<dyn TokenDecoder<B>> {
        let eot = self.tokenizer.special_token(SpecialToken::EndofText).unwrap();
        let decoder: GreedyDecoder<B> = GreedyDecoder::new(self.decode_options.temperature, eot as i32);
        return Box::new(decoder);
    }


    fn get_initial_tokens(&self) -> Vec<usize> {
        let transcription_token = self.tokenizer.special_token(SpecialToken::Transcribe).unwrap();
        let start_of_prev_token = self.tokenizer.special_token(SpecialToken::StartofPrev).unwrap();
        let lang_token = self.tokenizer.special_token(SpecialToken::Language(Language::Chinese)).unwrap();
        let first_timestamp_token = self.tokenizer.special_token(SpecialToken::Timestamp(0.0)).unwrap();
        let notimestamp = self.tokenizer.special_token(SpecialToken::NoTimeStamps).unwrap();
        let sot_token = self.tokenizer.special_token(SpecialToken::StartofTranscript).unwrap();
        return vec![sot_token, lang_token, transcription_token, notimestamp];
    }


    pub fn run(&self, mels: Tensor<B, 3>) -> Vec<DecodingResult<B>> {
        let [n_batch, n_mel, n_ctx] = mels.dims();
        println!("mel info: n_batch = {n_batch}, n_mel = {n_mel}, n_ctx = {n_ctx}");

        let device = mels.device();
        let audio_features = self.model.forward_encoder(mels);

        let n_ctx_max_encoder = self.model.encoder_ctx_size();
        let n_ctx_max_decoder = self.model.decoder_ctx_size();

        let padding = 0;
        if n_ctx + padding > n_ctx_max_encoder {
            println!(
                "Audio has length of {} which exceeds maximum length {}. It will be clipped.",
                n_ctx + padding,
                n_ctx_max_encoder
            );
        }

        let start_token = self.tokenizer.special_token(SpecialToken::StartofTranscript).unwrap();
        let transcription_token = self.tokenizer.special_token(SpecialToken::Transcribe).unwrap();
        let start_of_prev_token = self.tokenizer.special_token(SpecialToken::StartofPrev).unwrap();
        // let lang_token = self.tokenizer.special_token(SpecialToken::Language(lang)).unwrap();
        let first_timestamp_token = self.tokenizer.special_token(SpecialToken::Timestamp(0.0)).unwrap();
        let end_token = self.tokenizer.special_token(SpecialToken::EndofText).unwrap();
        // let notimestamp = self.tokenizer.special_token(SpecialToken::NoTimeStamps).unwrap();
        let decoder = self.init_decoder();
        //
        // // including the prev tokens causes whisper to hallucinate, repeating itself and failing to determine end of text
        // /*let mut tokens: Vec<usize> = iter::once(start_of_prev_token)
        // .chain(prev_normal_tokens.into_iter().cloned())
        // .chain(iter::once(start_token))
        // .chain(iter::once(transcription_token))
        // .chain(iter::once(bpe.special_token(SpecialToken::Timestamp(0.0)).unwrap()))
        // .collect();*/
        //
        let initial_tokens: Vec<i32> = self.get_initial_tokens().into_iter().map(|x| x as i32).collect();
        let sample_begin = initial_tokens.len();


        let mut logit_filters = Vec::new();
        if self.decode_options.suppress_blank {
            logit_filters.push(Box::new(SuppressBlank::new(self.tokenizer.clone(), sample_begin)));
        }


        let mut tokens: Tensor<B, 2, Int> = Tensor::from_ints(
            &*initial_tokens,
            &device,
        ).reshape([1, sample_begin]).repeat(0, n_batch);

        println!("initial_tokens = {tokens}");

        let mut sum_logprobs = Tensor::<B, 2>::zeros([n_batch, 1], &device);
        let no_speech_probs = Tensor::<B, 2>::full([n_batch, 1], f32::INFINITY, &device);
        for i in 0..self.decode_options.sample_len.unwrap_or(0) {
            let logits = self.model.forward_decoder(tokens.clone(), audio_features.clone());
            if i == 0 && self.tokenizer.special_token(SpecialToken::NoSpeech).is_some() {
                // save no_speech_probs
                // no_speech_probs
            }
            let [n_audio, n_group, vol_size] = logits.dims();
            let mut logits = logits.slice([0..n_batch, (n_group - 1)..n_group]).reshape([n_audio, vol_size]);

            for filter in &logit_filters {
                logits = filter.apply(logits, &tokens);
            }
            let (next_tokens, completed) = decoder.update(tokens, logits, &mut sum_logprobs);
            println!("next_tokens = {next_tokens}, completed = {completed}");
            tokens = next_tokens;
        }

        //
        // type BeamNode = beam::BeamNode<BeamSearchToken>;
        //
        // let initial_tokens = BeamNode {
        //     seq: initial_tokens,
        //     log_prob: 0.0,
        // };
        //
        // let encoder_output = whisper.forward_encoder(mels);
        //
        // let neg_infty = -f32::INFINITY;
        // /*let mut nonspecial_mask: Vec<f32> = (0..bpe.vocab_size()).into_iter().map(|tok| /*if bpe.is_special(tok) {neg_infty} else {0.0}*/ 0.0).collect();
        // //nonspecial_mask[end_token] = neg_infty;
        // let nonspecial_mask = Tensor::from_floats(Data::new(
        //     nonspecial_mask,
        //     [bpe.vocab_size()].into(),
        // )).to_device(&device);*/
        //
        // let beam_size = 5;
        // let max_depth = 100;
        //
        // let beamsearch_is_finished = |toks: &[BeamSearchToken]| {
        //     if let Some(btok) = toks.last() {
        //         btok.token == end_token
        //     } else {
        //         false
        //     }
        // };
        //
        // let vocab_size = bpe.vocab_size();
        // let mut special_tokens_maskout: Vec<f32> = (0..vocab_size).into_iter().map(|token| if bpe.is_special(token) { neg_infty } else { 0.0 }).collect();
        // //special_tokens_maskout[end_token] = 1.0;
        //
        // let special_tokens_maskout = Tensor::from_data(Data::new(
        //     special_tokens_maskout,
        //     [vocab_size].into(),
        // ).convert(), &device)
        //     .to_device(&device);
        //
        // let beamsearch_next = |beams: &[BeamNode]| {
        //     // convert tokens into tensor
        //     let max_seq_len = beams.iter().map(|beam| beam.seq.len()).max().unwrap_or(0);
        //     let flattened_tokens: Vec<_> = beams
        //         .iter()
        //         .flat_map(|beam| {
        //             let additional_tokens = max_seq_len - beam.seq.len();
        //             beam.seq.iter().map(|btok| btok.token).chain(iter::once(0).cycle().take(additional_tokens))
        //         })
        //         .collect();
        //
        //     let token_tensor = Tensor::from_ints(Data::from_usize(Data::new(
        //         flattened_tokens,
        //         [beams.len(), max_seq_len].into(),
        //     )), &device)
        //         .to_device(&device);
        //
        //     let logits = self.model.forward_decoder(token_tensor, encoder_output.clone().repeat(0, beams.len()));
        //     let logits = if max_seq_len > 5 {
        //         logits
        //     } else {
        //         logits + special_tokens_maskout.clone().unsqueeze().slice([0..1, 0..1, 0..51864])
        //     };
        //     let log_probs = log_softmax(logits, 2);
        //
        //     let [n_batch, n_token, n_dict] = log_probs.dims();
        //     let beam_log_probs = beams.iter().enumerate().map(|(i, beam)| {
        //         let batch = i;
        //         let token_index = beam.seq.len() - 1;
        //
        //         log_probs.clone().slice([batch..batch + 1, token_index..token_index + 1]).flatten::<1>(0, 2).into_data().value
        //     });
        //
        //     let continuations = beam_log_probs
        //         .zip(beams)
        //         .map(|(log_probs, beam)| {
        //             log_probs
        //                 .into_iter()
        //                 .map(|log_prob| log_prob.elem::<f64>())
        //                 .enumerate()
        //                 .map(|(token_id, log_prob)| {
        //                     (
        //                         BeamSearchToken {
        //                             token: token_id,
        //                             log_prob: log_prob,
        //                         },
        //                         beam.log_prob + log_prob,
        //                     )
        //                 }
        //                 )
        //                 .collect()
        //         }).collect();
        //
        //     continuations
        // };
        //
        // let tokens: Vec<_> = beam::beam_search(vec![initial_tokens], beamsearch_next, beamsearch_is_finished, beam_size, max_depth)
        //     .into_iter()
        //     .map(|btok| btok.token)
        //     .collect();
        //
        // /*let mut tokens: Vec<_> = [start_token, lang_token, transcription_token, notimestamp].to_vec();
        //
        // loop {
        //     if tokens.len() >= n_ctx_max_decoder {
        //         tokens.push(end_token);
        //         break;
        //     }
        //
        //     let token_tensor = Tensor::from_ints(Data::from_usize(Data::new(
        //         tokens.clone(),
        //         [tokens.len()].into(),
        //     )))
        //     .unsqueeze::<2>()
        //     .to_device(&device);
        //
        //     let out = whisper.forward_decoder(token_tensor, encoder_output.clone());
        //
        //     let [n_batch, n_token, n_dict] = out.dims();
        //     let last_row: Tensor<B, 1> = out.slice([0..1, (n_token - 1)..n_token]).flatten(0, 2);
        //
        //     let token_id = last_row.clone().argmax(0).into_scalar().to_usize().unwrap();
        //     let token_logit = last_row
        //         .clone()
        //         .slice([token_id..(token_id + 1)])
        //         .into_scalar()
        //         .to_f64()
        //         .unwrap();
        //     let eot_logit = last_row
        //         .slice([end_token..(end_token + 1)])
        //         .into_scalar()
        //         .to_f64()
        //         .unwrap();
        //
        //     tokens.push(token_id);
        //     //println!("{}", bpe.decode(&[token_id], false)?);
        //
        //     // if end of text confidence is great enough then stop
        //     if (eot_logit - token_logit).exp() > 0.5 {
        //         if token_id != end_token {
        //             tokens.push(end_token);
        //         }
        //         break;
        //     }
        //
        //     let repeat_window_size = 5;
        //     let min_n_repeats = 4; // three times to charm, four to scorn
        //     /*println!("{}", bpe.decode(&tokens[..], false).unwrap());
        //     if let Some(period) = repetition_period(&tokens[..], min_n_repeats) {
        //         println!("period = {}", period);
        //         let end = first_repetition_end(&tokens[..], period);
        //
        //         tokens.truncate(end);
        //         tokens.push(end_token);
        //         break;
        //     }*/
        //     if let Some( (index_of_first_repeat, end) ) =
        //         find_repeated_tokens_index(&tokens[..], repeat_window_size, min_n_repeats)
        //     {
        //         //let end = index_of_first_repeat + repeat_window_size;
        //
        //         tokens.truncate(end);
        //         tokens.push(end_token);
        //         break;
        //     }
        // }*/
        //
        // let text = bpe.decode(&tokens[..], true)?;
        //
        // return Ok((text, tokens));
        vec![]
    }
}

#[cfg(test)]
mod test {
    use burn_wgpu::{Wgpu, AutoGraphicsApi, WgpuDevice};
    use crate::whsiper_helper::{WhichModel, WhisperHelper};
    use tokio;

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

        let M: WhisperHelper<CurBackend> = WhisperHelper::new(WhichModel::Base, &device).await;
    }
}
