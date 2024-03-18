use burn::tensor::backend::Backend;

use burn::tensor::Tensor;
use num_traits::Zero;
use reqwest::{self, IntoUrl};

use std::fs::File;

use std::io::{copy, Write};

use std::path::{Path, PathBuf};

use whisper::model::{Whisper, WhisperConfig};

use whisper::token::{Gpt2Tokenizer, Language, SpecialToken};

use burn::record::Recorder;

async fn download_from_url_to_file<
    T: IntoUrl + std::fmt::Debug,
    P: AsRef<Path> + std::fmt::Debug,
>(
    url: T,
    load_file_path: P,
) -> Result<(), reqwest::Error> {
    println!("download fomr [{url:?}] \n\t to {load_file_path:?}");
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
            let url =  match self{
                Self::TinyEn => "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
                Self::Tiny=> r"https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
                Self::BaseEn=> r"https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
                Self::Base=> r"https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
                Self::SmallEn=> r"https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
                Self::Small=> r"https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
                Self::MediumEn=> r"https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
                Self::Medium=> r"https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
                Self::LargeV1=> r"https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large-v1.pt",
                Self::LargeV2=> r"https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt",
                Self::LargeV3=> r"https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
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

pub struct WhisperHelper<B: Backend> {
    pub config: WhisperConfig,
    pub model: Whisper<B>,
    pub tokenizer: Gpt2Tokenizer,
    pub kind: WhichModel,
}

impl<B: Backend> WhisperHelper<B> {
    pub async fn new(model_kind: WhichModel, device: &B::Device) -> WhisperHelper<B> {
        let model_dir = Path::new("./model");
        model_kind.download_to_dir(model_dir).await.expect("msg");
        let config_path = model_dir.join(model_kind.config_path());
        let config = <WhisperConfig as burn::config::Config>::load(config_path).expect("msg");
        let load_args = burn_import::pytorch::LoadArgs::new(
            model_dir.join(model_kind.model_local_path()).into(),
        )
        .with_debug_print()
        .with_key_remap("mlp.0", "mlp0")
        .with_key_remap("mlp.2", "mlp2")
        .with_top_level_key("model_state_dict");
        let model =
            burn_import::pytorch::PyTorchFileRecorder::<burn::record::HalfPrecisionSettings>::new()
                .load(load_args, device)
                .map(|record| burn::module::Module::load_record(config.init(device), record))
                .expect("msg");

        let tokenizer = Gpt2Tokenizer::new(model_dir.join(model_kind.tokenizer_json_path())).expect("msg");

        Self {
            config,
            model,
            tokenizer,
            kind: model_kind,
        }
    }

    /// Returns the token id for the selected language.

    pub fn detect_language(&self, mel: &Tensor<B, 3>) -> (Language, f32) {
        if !self.kind.is_multilingual() {
            return (Language::English, 1.0f32);
        }
        let mel = whisper::audio::pad_or_trim(mel, whisper::audio::N_FRAMES);
        let [_bsize, _, seq_len] = mel.dims();
        println!("before mel dims = {:?}", mel.dims());
        let mel = self.model.forward_encoder(mel.clone());
        println!("after mel dims = {:?}", mel.dims());

        // let mel = mel.clone().narrow(
        //     2,
        //     0,
        //     usize::min(seq_len, model.config().max_source_positions),
        // )?;

        let device = &mel.device();

        println!("vocab_size = {}", self.tokenizer.vocab_size());
        let language_token_ids = whisper::token::LANGUAGES
            .iter()
            .map(|t| {
                // println!("{t}");

                self.tokenizer
                    .special_token(SpecialToken::Language(Language::from_str(t).unwrap()))
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let sot_token = self
            .tokenizer
            .special_token(SpecialToken::StartofTranscript)
            .unwrap();

        let n_audio = mel.dims()[0];

        let x = Tensor::<B, 2, burn::tensor::Int>::full([n_audio, 1], sot_token as i32, device);

        let logits = self.model.forward_decoder(x, mel);

        //

        let logits_dims = logits.dims();

        println!("logits_dims = {logits_dims:?}");

        let mask = Tensor::<B, 1>::ones([logits_dims[2]], device);

        for l_token_id in language_token_ids {
            mask.to_data().value[l_token_id] = B::FloatElem::zero();
        }

        let mask = mask
            .unsqueeze::<2>()
            .repeat(0, logits_dims[0] * logits_dims[1])
            .reshape(logits_dims);

        println!("mask dims = {:?}", mask.dims());
        let mask = mask.equal(Tensor::<B, 3>::ones(logits_dims, device));
        let logits = logits.mask_fill(mask, -f32::INFINITY);
        println!("after mask logits dims = {:?}", logits.dims());
        let language_tokens = logits.clone().argmax(2); // language_tokens = logits.argmax(dim=-1)
        println!("language_tokens = {language_tokens:?}");
        let language_token_probs = burn::tensor::activation::softmax(logits, 2); // dim= -1
        let (a, b) = language_token_probs.max_dim_with_indices(2);
        println!("{:} b = {:?}", a, b);

        return (Language::English, 1.0f32);
    }
}
