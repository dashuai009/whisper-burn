use burn::tensor::{activation::relu, backend::Backend, ElementConversion, Tensor};

use crate::helper::*;

pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 80;
pub const WINDOW_LENGTH: usize = N_FFT;
pub const CHUNK_LENGTH: usize = 30;
pub const SAMPLE_RATE: usize = 16000;
pub const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE;
//  # 480000 samples in a 30-second chunk
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH;
// 3000 frames in a mel spectrogram input
pub const N_SAMPLES_PER_TOKEN: usize = HOP_LENGTH * 2;
//   # the initial convolutions has stride 2
pub const FRAMES_PER_SECOND: usize = SAMPLE_RATE / HOP_LENGTH;
//  # 10ms per audio frame
pub const TOKENS_PER_SECOND: usize = SAMPLE_RATE / N_SAMPLES_PER_TOKEN; //  # 20ms per audio token

/// Returns the maximum number of waveform samples that can be submitted to `prep_audio`
/// without receiving more than `n_frame_max` frames.
pub fn max_waveform_samples(n_frame_max: usize) -> usize {
    // number of waveform samples must be less than this
    let n_samples_max = HOP_LENGTH * (n_frame_max + 1) + is_odd(N_FFT) as usize;

    n_samples_max - 1
}

fn is_odd(x: usize) -> bool {
    if x % 2 == 0 {
        false
    } else {
        true
    }
}

/// Transform an input waveform into a format interpretable by Whisper.
/// With a waveform size of (n_batch, n_samples) the output will be of size (n_batch, n_mels, n_frame)
/// where n_mels = 80,
/// n_frame = int( ( n_samples_padded - n_fft ) / hop_length ),
/// n_samples_padded = if n_fft is even: n_samples + n_fft else: n_samples + n_fft - 1,
/// n_fft = 400,
/// hop_length = 160.
pub fn log_mel_spectrogram<B: Backend>(audio: Tensor<B, 2>) -> Tensor<B, 3> {
    let device = audio.device();

    let window = hann_window_device(WINDOW_LENGTH, &device);
    let (stft_real, stft_imag) = stfft(audio, N_FFT, HOP_LENGTH, window);

    let magnitudes = stft_real.powf_scalar(2.0) + stft_imag.powf_scalar(2.0);
    let [n_batch, n_row, n_col] = magnitudes.dims();
    let magnitudes = magnitudes.slice([0..n_batch, 0..n_row, 0..(n_col - 1)]);

    let mel_spec = get_mel_filters_device(SAMPLE_RATE as f64, N_FFT, N_MELS, false, &device)
        .unsqueeze()
        .matmul(magnitudes);

    let log_spec = mel_spec.clamp(1.0e-10, f32::MAX).log() / (10.0f64).ln();

    let max: f64 = log_spec.clone().max().into_scalar().elem();

    let log_spec = tensor_max_scalar(log_spec, max - 8.0);
    let log_spec = (log_spec + 4.0) / 4.0;

    return log_spec;
}

fn get_mel_filters<B: Backend>(
    sample_rate: f64,
    n_fft: usize,
    n_mels: usize,
    htk: bool,
) -> Tensor<B, 2> {
    get_mel_filters_device(sample_rate, n_fft, n_mels, htk, &B::Device::default())
}

/// mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
/// https://librosa.org/doc/main/_modules/librosa/filters.html#mel
fn get_mel_filters_device<B: Backend>(
    sample_rate: f64,
    n_fft: usize,
    n_mels: usize,
    htk: bool,
    device: &B::Device,
) -> Tensor<B, 2> {
    let fmin = 0.0;
    let fmax = sample_rate * 0.5;

    //let weights = Tensor::zeros([n_mels, 1 + n_fft / 2]);

    // Center freqs of each FFT bin
    let fftfreqs = fft_frequencies_device(sample_rate, n_fft, device);
    let [n_ffefreqs] = fftfreqs.dims();

    // 'Center freqs' of mel bands - uniformly spaced between limits
    let mel_f_size = n_mels + 2;
    let mel_f = mel_frequencies_device(mel_f_size, fmin, fmax, htk, device);

    // fdiff = np.diff(mel_f)
    let fdiff = mel_f.clone().slice([1..mel_f_size]) - mel_f.clone().slice([0..(mel_f_size - 1)]);

    // ramps = np.subtract.outer(mel_f, fftfreqs)
    let ramps = mel_f
        .clone()
        .unsqueeze::<2>()
        .transpose()
        .repeat(1, n_ffefreqs)
        - fftfreqs.unsqueeze();

    /*for i in range(n_mels):
    # lower and upper slopes for all bins
    lower = -ramps[i] / fdiff[i]
    upper = ramps[i + 2] / fdiff[i + 1]

    # .. then intersect them with each other and zero
    weights[i] = np.maximum(0, np.minimum(lower, upper))*/

    // lower and upper slopes for all bins
    let lower = -ramps.clone().slice([0..n_mels])
        / fdiff
        .clone()
        .slice([0..n_mels])
        .unsqueeze::<2>()
        .transpose();
    let upper = ramps.slice([2..(2 + n_mels)])
        / fdiff.slice([1..(1 + n_mels)]).unsqueeze::<2>().transpose();

    // .. then intersect them with each other and zero
    let weights = relu(tensor_min(lower, upper));

    // Slaney-style mel is scaled to be approx constant energy per channel
    //enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    let enorm = (mel_f.clone().slice([2..(n_mels + 2)]) - mel_f.clone().slice([0..n_mels]))
        .powf_scalar(-1.0)
        * 2.0;
    //weights *= enorm[:, np.newaxis]
    let weights = weights * enorm.unsqueeze::<2>().transpose();

    // Only check weights if f_mel[0] is positive
    /*if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
    # This means we have an empty channel somewhere
    warnings.warn(
        "Empty filters detected in mel frequency basis. "
        "Some channels will produce empty responses. "
        "Try increasing your sampling rate (and fmax) or "
        "reducing n_mels.",
        stacklevel=2,
    )*/
    if !(all_zeros(mel_f.slice([0..(n_mels - 2)])) || all_zeros(relu(-weights.clone().max_dim(1))))
    {
        println!("Empty filters detected in mel frequency basis. \nSome channels will produce empty responses. \nTry increasing your sampling rate (and fmax) or reducing n_mels.");
    }

    return weights;
}

fn fft_frequencies_device<B: Backend>(
    sample_rate: f64,
    n_fft: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    //return np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    Tensor::arange(0..((n_fft as i64) / 2 + 1), device)
        .float()
        .mul_scalar(sample_rate / n_fft as f64)
}

fn mel_frequencies_device<B: Backend>(
    n_mels: usize,
    fmin: f64,
    fmax: f64,
    htk: bool,
    device: &B::Device,
) -> Tensor<B, 1> {
    // 'Center freqs' of mel bands - uniformly spaced between limits
    let min_mel = hz_to_mel(fmin, htk);
    let max_mel = hz_to_mel(fmax, htk);

    //mels = np.linspace(min_mel, max_mel, n_mels)
    let mels = Tensor::arange(0..(n_mels as i64), device)
        .float()
        .mul_scalar((max_mel - min_mel) / (n_mels - 1) as f64)
        .add_scalar(min_mel);

    //hz: np.ndarray = mel_to_hz(mels, htk=htk)
    mel_to_hz_tensor(mels, htk)
}

fn hz_to_mel(freq: f64, htk: bool) -> f64 {
    if htk {
        return 2595.0 * (1.0 + freq / 700.0).log10();
    }

    // Fill in the linear part
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;

    //let mel = (freq - f_min) / f_sp;

    // Fill in the log-scale part

    let min_log_hz = 1000.0; // beginning of log region (Hz)
    let min_log_mel = (min_log_hz - f_min) / f_sp; // same (Mels)
    let logstep = (6.4f64).ln() / 27.0; // step size for log region

    /*if frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep*/

    let mel = if freq >= min_log_hz {
        min_log_mel + (freq / min_log_hz).ln() / logstep
    } else {
        (freq - f_min) / f_sp
    };

    return mel;
}

fn mel_to_hz_tensor<B: Backend>(mel: Tensor<B, 1>, htk: bool) -> Tensor<B, 1> {
    if htk {
        return (_10pow(mel / 2595.0) - 1.0) * 700.0;
    }

    // Fill in the linear scale
    let f_min = 0.0;
    let f_sp = 200.0 / 3.0;
    //let freqs = f_min + f_sp * mel;

    // And now the nonlinear scale
    let min_log_hz = 1000.0; // beginning of log region (Hz)
    let min_log_mel = (min_log_hz - f_min) / f_sp; // same (Mels)
    let logstep = (6.4f64).ln() / 27.0; // step size for log region

    /*if mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))*/

    let log_t = mel.clone().greater_equal_elem(min_log_mel).float();
    let freq = log_t.clone() * (((mel.clone() - min_log_mel) * logstep).exp() * min_log_hz)
        + (-log_t + 1.0) * (mel * f_sp + f_min);

    /*let freq = if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_min + f_sp * mel
    };*/

    return freq;
}

pub fn hann_window_device<B: Backend>(window_length: usize, device: &B::Device) -> Tensor<B, 1> {
    Tensor::arange(0..(window_length as i64), device)
        .float()
        .mul_scalar(std::f64::consts::PI / window_length as f64)
        .sin()
        .powf_scalar(2.0)
}

/// Short time Fourier transform that takes a waveform input of size (n_batch, n_sample) and returns (real_part, imaginary_part) frequency spectrums.
/// The size of each returned tensor is (n_batch, n_freq, n_frame)
/// where n_freq = int(n_fft / 2 + 1), n_frame = int( ( n_sample_padded - n_fft ) / hop_length ) + 1,
/// n_sample_padded = if n_fft is even: n_sample + n_fft else: n_sample + n_fft - 1.
pub fn stfft<B: Backend>(
    input: Tensor<B, 2>,
    n_fft: usize,
    hop_length: usize,
    window: Tensor<B, 1>,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    let [n_batch, orig_input_size] = input.dims();

    assert!(orig_input_size >= n_fft);

    let device = input.device();

    // add reflection padding to center the windows on the input times
    let pad = n_fft / 2;
    let left_pad = reverse(input.clone().slice([0..n_batch, 1..(pad + 1)]), 1);
    let right_pad = reverse(
        input.clone().slice([
            0..n_batch,
            (orig_input_size - pad - 1)..(orig_input_size - 1),
        ]),
        1,
    );
    let input = Tensor::cat(vec![left_pad, input, right_pad], 1);

    // pad window to length n_fft
    let [orig_window_length] = window.dims();
    let window = if orig_window_length < n_fft {
        let left_pad = (n_fft - orig_window_length) / 2;
        let right_pad = n_fft - orig_window_length - left_pad;
        Tensor::cat(
            vec![
                Tensor::zeros([left_pad], &device),
                window,
                Tensor::zeros([right_pad], &device),
            ],
            0,
        )
    } else {
        window
    };

    let [_, input_size] = input.dims();

    let n_frame = (input_size - n_fft) / hop_length + 1;
    let n_freq = n_fft / 2 + 1; // assuming real input there is conjugate symmetry

    // construct matrix of overlapping input windows
    let num_parts = div_roundup(n_fft, hop_length);
    let n_hops = div_roundup(input_size, hop_length);
    let padded_input_size = n_hops * hop_length;
    let padding = Tensor::zeros([n_batch, padded_input_size - input_size], &device);
    let template = Tensor::cat(vec![input, padding], 1)
        .reshape([n_batch, n_hops, hop_length])
        .transpose();
    let parts: Vec<_> = (0..num_parts)
        .into_iter()
        .map(|i| {
            template
                .clone()
                .slice([0..n_batch, 0..hop_length, i..(n_frame + i)])
        })
        .collect();
    let input_windows = Tensor::cat(parts, 1).slice([0..n_batch, 0..n_fft, 0..n_frame]);

    // construct matrix of wave angles
    let coe = std::f64::consts::PI * 2.0 / n_fft as f64;
    let b = Tensor::arange(0..(n_freq as i64), &device)
        .float()
        .mul_scalar(coe)
        .unsqueeze::<2>()
        .transpose()
        .repeat(1, n_fft)
        * Tensor::arange(0..(n_fft as i64), &device)
        .float()
        .unsqueeze::<2>();

    // convolve the input slices with the window and waves
    let real_part = (b.clone().cos() * window.clone().unsqueeze())
        .unsqueeze()
        .matmul(input_windows.clone());
    let imaginary_part = (b.sin() * (-window).unsqueeze())
        .unsqueeze()
        .matmul(input_windows);

    return (real_part, imaginary_part);
}

fn div_roundup(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

pub fn pad_or_trim<const D: usize, B: Backend>(
    input_tensor: &Tensor<B, D>,
    length: usize,
) -> Tensor<B, D> {
    let device = input_tensor.device();
    let dims = input_tensor.dims();
    if dims[D - 1] > length {
        let mut new_dims = dims.clone();
        new_dims[D - 1] = length;
        let mut ranges = vec![];
        for d in new_dims {
            ranges.push(0..d)
        }
        input_tensor.clone().slice::<D>(ranges.try_into().unwrap())
    } else if dims[D - 1] == length {
        input_tensor.clone()
    } else {
        let mut padding_dims = dims.clone();
        padding_dims[D - 1] = length - padding_dims[D - 1];
        let padding = Tensor::<B, D>::zeros(padding_dims, &device);
        Tensor::<B, D>::cat(vec![input_tensor.clone(), padding], D - 1)
    }
}

#[cfg(test)]
mod test {
    use burn::tensor::backend::Backend;
    use burn::tensor::Tensor;

    use crate::audio::{mel_frequencies_device, pad_or_trim};

    // #[test]
    // fn test_fft_frequencies() {
    //     // let sr = 1000.0; // stating sample rate
    //     // let n_fft = 100; // stating the window size of fft
    //     // let fftfreqs = fft_frequencies::<B>(sr, n_fft);
    //     // println!("{:?}", fftfreqs);
    // }
    //
    // fn test_mel_frequencies<B: Backend>(htk: bool) {
    //     // let n_mels = 128; // stating the number of Mel bands
    //     // let fmin = 0.0; // stating the lowest frequency
    //     // let fmax = 22050.0; // stating the highest frequency
    //     // let melfreqs = mel_frequencies::<B>(n_mels + 2, fmin, fmax, htk);
    //     // println!("{:?}", melfreqs);
    // }

    #[test]
    fn test_pad_or_trim() {
        cfg_if::cfg_if! {
            if #[cfg(feature = "wgpu-backend")] {
                type CurBackend = burn_wgpu::Wgpu<burn_wgpu::AutoGraphicsApi, f32, i32>;
                let device = burn_wgpu::WgpuDevice::BestAvailable;
            } else if #[cfg(feature = "torch-backend")] {
                type CurBackend = LibTorch<f32>;
                let device = LibTorchDevice::Cuda(0);
            }
        }


        let input = Tensor::<CurBackend, 1>::from_data(
            [1.0, 2.0, 3.0],
            &device,
        );
        let output = pad_or_trim(&input, 5);
        let expected =
            Tensor::<CurBackend, 1>::from_data(
                [1.0, 2.0, 3.0, 0.0, 0.0],
                &device,
            );
        output.to_data().assert_approx_eq(
            &expected.to_data(), 7,
        );


        let input1 = Tensor::<CurBackend, 2>::from_data(
            [
                [1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0]
            ],
            &device,
        );
        let output1 = pad_or_trim(&input1, 3);
        let expected1 =
            Tensor::<CurBackend, 2>::from_data(
                [
                    [1.0, 2.0, 3.0],
                    [1.0, 2.0, 3.0],
                ],
                &device,
            );
        output1.to_data().assert_approx_eq(
            &expected1.to_data(), 7,
        );
    }
}
