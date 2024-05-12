use burn::prelude::{Backend, Int, Tensor};
use burn::tensor::activation::log_softmax;

pub trait TokenDecoder<B: Backend> {
    fn reset(&mut self);

    /// Specify how to select the next token, based on the current trace and logits
    /// ## Args
    /// - `tokens` : Tensor, shape = (n_batch, current_sequence_length)
    ///             all tokens in the context so far, including the prefix and sot_sequence tokens
    /// - `logits` : Tensor, shape = (n_batch, vocab_size)
    ///             per-token logits of the probability distribution at the current step
    ///
    /// - `sum_logprobs` : Tensor, shape = (n_batch, 1)
    ///             cumulative log probabilities for each sequence
    ///
    /// ## Returns
    /// - `tokens` : Tensor, shape = (n_batch, current_sequence_length + 1)
    //             the tokens, appended with the selected next token
    ///
    /// - `completed` : bool
    ///             True if all sequences has reached the end of text
    fn update(&self, tokens: Tensor<B, 2, Int>, logits: Tensor<B, 2>, sum_logprobs: &mut Tensor<B, 2>) -> (Tensor<B, 2, Int>, bool);

    /// Finalize search and return the final candidate sequences
    ///
    /// ## Parameters
    /// - `tokens` : Tensor, shape = (n_audio, n_group, current_sequence_length)
    ///            all tokens in the context so far, including the prefix and sot_sequence
    ///
    /// - `sum_logprobs` : Tensor, shape = (n_audio, n_group)
    ///             cumulative log probabilities for each sequence
    ///
    ///  ## Returns
    /// - `tokens` : Sequence[Sequence[Tensor]], length = n_audio
    ///             sequence of Tensors containing candidate token sequences, for each audio input
    ///
    /// - `sum_logprobs` : List[List[float]], length = n_audio
    ///             sequence of cumulative log probabilities corresponding to the above
    ///
    fn finalize(&self, tokens: Tensor<B, 3, Int>, sum_logprobs: Tensor<B, 2>)
                -> (Tensor<B, 3, Int>, Tensor<B, 2>);
}


pub struct GreedyDecoder {
    temperature: f32,
    eot: i32
}

impl GreedyDecoder {
    pub fn new(temperature: f32, eot: i32) -> GreedyDecoder {
        GreedyDecoder {
            temperature,
            eot
        }
    }
}

impl<B: Backend> TokenDecoder<B> for GreedyDecoder {
    fn reset(&mut self) {}

    fn update(&self, tokens: Tensor<B, 2, Int>, logits: Tensor<B, 2>, sum_logprobs: &mut Tensor<B, 2>) -> (Tensor<B, 2, Int>, bool) {
        // if self.temperature == 0:
        //     next_tokens = logits.argmax(dim=-1)
        // else:
        //     next_tokens = Categorical(logits=logits / self.temperature).sample()
        //
        // logprobs = F.log_softmax(logits.float(), dim=-1)
        // current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        // sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)
        //
        // next_tokens[tokens[:, -1] == self.eot] = self.eot
        // tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)
        //
        // completed = (tokens[:, -1] == self.eot).all()
        // return tokens, completed

        let device = tokens.device();
        let [_, cur_seq_len] = tokens.dims();
        let [n_audio, vol_size] = logits.dims();

        // next_tokens: [n_audio, 1]
        let next_tokens = if self.temperature == 0.0 {
            logits.clone().argmax(1) // dim == -1 == 1
        } else {
            logits.clone().argmax(1)
        };
        // println!("next_tokens = {next_tokens}");

        // [n_audio, val_size]
        let logprobs = log_softmax(logits, 1);


        let current_logprobs = logprobs
            // .select()
            .gather(1, next_tokens.clone().repeat(1, vol_size))
            .slice([0..n_audio, 0..1]);

        // println!("cur_logprobs = {current_logprobs}");

        let all_eot = Tensor::full([n_audio, 1], self.eot, &device);
        let mask_tokens = tokens.clone()
            .slice([0..n_audio, (cur_seq_len - 1)..cur_seq_len])
            .equal(all_eot.clone());
        // println!("mask_tokens = {mask_tokens}");

        *sum_logprobs = (sum_logprobs.clone()).add(current_logprobs.mask_fill(mask_tokens.clone(), 0.0));
        // println!("sum_logprobs = {sum_logprobs}");

        let next_tokens = next_tokens.mask_fill(mask_tokens.clone(), self.eot);
        // println!("next_tokens = {next_tokens}");

        let tokens = Tensor::cat(vec![tokens, next_tokens.clone()], 1);// dim = -1
        let completed = next_tokens.equal(all_eot).all().into_scalar();
        (tokens, completed)
    }

    fn finalize(&self, tokens: Tensor<B, 3, Int>, sum_logprobs: Tensor<B, 2>) -> (Tensor<B, 3, Int>, Tensor<B, 2>) {
        // let tokens = tokens.pad((0, 0, 1, 1), self.eot.into());
        (tokens, sum_logprobs)
    }
}