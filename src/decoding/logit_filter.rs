use std::rc::Rc;
use burn::prelude::{Backend, Bool, Data, Int, Tensor};
use tokenizers::Tokenizer;
use crate::token::{Gpt2Tokenizer, SpecialToken};

/// Apply any filtering or masking to logits in-place
/// ## Parameters
/// - `logits` : Tensor, shape = (n_batch, vocab_size)
///    per-token logits of the probability distribution at the current step
/// - `tokens` : Tensor, shape = (n_batch, current_sequence_length)
///    all tokens in the context so far, including the prefix and sot_sequence tokens
/// ## return
/// - `new_logits`: Tensor, shape = (n_batch, vocab_size)
pub trait LogitFilter<B: Backend> {
    fn apply(&self, logits: Tensor<B, 2>, tokens: &Tensor<B, 2, Int>) -> Tensor<B, 2>;
}


// SuppressBlank implementation
pub struct SuppressBlank {
    tokenizer: Rc<Gpt2Tokenizer>,
    sample_begin: usize,
}

impl SuppressBlank {
    pub fn new(tokenizer: Rc<Gpt2Tokenizer>, sample_begin: usize) -> Self {
        SuppressBlank {
            tokenizer,
            sample_begin,
        }
    }
}

impl<B: Backend> LogitFilter<B> for SuppressBlank {
    fn apply(&self, logits: Tensor<B, 2>, tokens: &Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let device = logits.device();
        let [n_batch, vocab_size] = logits.dims();
        let [_, seq_length] = tokens.dims();
        return if seq_length == self.sample_begin {
            // Collect indices to suppress (space and EOT)
            let mut suppress_indices = vec![false; vocab_size];
            let  space_idx = self.tokenizer.encode(" ");
            suppress_indices[space_idx[0]] = true;

            suppress_indices[self.tokenizer.special_token(SpecialToken::EndofText).unwrap()] = true;

            let suppress_indices = Tensor::<B, 2, Bool>::from_data(
                Data::new(suppress_indices, [1, vocab_size].into()),
                &device,
            ).repeat(0, n_batch);
            logits.mask_fill(suppress_indices, f32::NEG_INFINITY)
        } else {
            logits
        };
    }
}


pub struct SuppressTokens{
    suppress_tokens: Vec<i32>
}

impl SuppressTokens{
    pub fn new(suppress_tokens: Vec<i32>) -> SuppressTokens{
        SuppressTokens{
            suppress_tokens,
        }
    }
}
impl<B:Backend> LogitFilter<B> for SuppressTokens{
    fn apply(&self, logits: Tensor<B, 2>, tokens: &Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let device = logits.device();
        let [n_batch, vocab_size] = logits.dims();
        let mut suppress_indices = vec![false; vocab_size];
        for token in self.suppress_tokens{
            suppress_indices[token] = true;
        }
        let suppress_indices = Tensor::<B, 2, Bool>::from_data(
            Data::new(suppress_indices, [1, vocab_size].into()),
            &device,
        ).repeat(0, n_batch);
        logits.mask_fill(suppress_indices, f32::NEG_INFINITY)
    }
}