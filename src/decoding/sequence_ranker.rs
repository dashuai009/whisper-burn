use burn::prelude::{Backend, Int};
use burn::tensor::Tensor;

pub trait SequenceRanker<B: Backend> {
    /// Given a list of groups of samples and their cumulative log probabilities,
    /// return the indices of the samples in each group to select as the final result
    ///
    /// ## Args
    /// - `tokens`: Tensor[n_batch, n_group, seq_len]
    /// - `sum_logprobs`: Tensor[n_batch, n_group]
    /// ## return
    /// Index[i]: 0 <= Index[i] < n_group, V.len() == n_batch
    fn rank(&self, tokens: &Vec<Vec<Vec<u32>>>, sum_logprobs:& Tensor<B, 2>) -> Vec<usize>;
}


pub struct TakeFirstGroup {}

impl TakeFirstGroup{
    pub fn new() -> TakeFirstGroup{
        TakeFirstGroup{}
    }
}

impl<B: Backend> SequenceRanker<B> for TakeFirstGroup {
    fn rank(&self, tokens:  &Vec<Vec<Vec<u32>>>, sum_logprobs:& Tensor<B, 2>) -> Vec<usize> {
       let n_batch = tokens.len();
        vec![0; n_batch]
    }
}