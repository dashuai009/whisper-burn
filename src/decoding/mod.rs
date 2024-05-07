mod token_decoder;
pub mod logit_filter;
pub mod sequence_ranker;

pub use token_decoder::TokenDecoder;
pub use token_decoder::GreedyDecoder;


use std::collections::HashSet;
use burn::prelude::Backend;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DecodingOptions {
    // whether to perform X->X "transcribe" or X->English "translate"
    pub task: String,

    // language that the audio is in; uses detected language if None
    pub language: Option<String>,

    // sampling-related options
    pub temperature: f32,
    pub sample_len: Option<usize>,  // maximum number of tokens to sample
    pub best_of: Option<usize>,     // number of independent sample trajectories, if t > 0
    pub beam_size: Option<usize>,   // number of beams in beam search, if t == 0
    pub patience: Option<f32>,      // patience in beam search

    // "alpha" in Google NMT, or None for length norm, when ranking generations
    pub length_penalty: Option<f32>,

    // text or tokens to feed as the prompt or the prefix
    pub prompt: Option<Prompt>,     // for the previous context
    pub prefix: Option<Prompt>,     // to prefix the current context

    // list of tokens ids to suppress
    pub suppress_tokens: Option<UserSuppressTokens>,
    pub suppress_blank: bool,       // this will suppress blank outputs

    // timestamp sampling options
    pub without_timestamps: bool,   // use to sample text tokens only
    pub max_initial_timestamp: Option<f32>,

    // implementation details
    pub fp16: bool,                 // use fp16 for most of the calculation
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum Prompt {
    Text(String),
    Tokens(Vec<i32>),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum UserSuppressTokens {
    Text(String),
    Tokens(HashSet<i32>),
}

impl Default for DecodingOptions {
    fn default() -> Self {
        Self {
            task: "transcribe".to_owned(),
            language: None,
            temperature: 0.0,
            sample_len: None,
            best_of: None,
            beam_size: None,
            patience: None,
            length_penalty: None,
            prompt: None,
            prefix: None,
            suppress_tokens: Some(UserSuppressTokens::Text("-1".to_owned())),
            suppress_blank: true,
            without_timestamps: false,
            max_initial_timestamp: Some(1.0),
            fp16: true,
        }
    }
}

