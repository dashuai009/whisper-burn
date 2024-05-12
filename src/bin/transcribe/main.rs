#[cfg(feature = "ffmpeg-input")]
extern crate ffmpeg_next;

mod whsiper_helper;


#[cfg(feature = "ffmpeg-input")]
use ffmpeg::{frame, media};
#[cfg(feature = "ffmpeg-input")]
use ffmpeg_next as ffmpeg;


cfg_if::cfg_if! {
    if #[cfg(feature = "wgpu-backend")] {
        use burn_wgpu::{Wgpu, WgpuDevice, AutoGraphicsApi};
    } else if #[cfg(feature = "torch-backend")] {
        use burn_tch::{LibTorch, LibTorchDevice};
    }
}

#[cfg(feature = "bound-input")]
use hound::{self, SampleFormat};

/// 从音视频文件中提取音频数据
/// 音频数据为16k Hz
/// //  cmd = [
//         "ffmpeg",
//         "-nostdin",
//         "-threads", "0",
//         "-i", file,
//         "-f", "s16le",
//         "-ac", "1",
//         "-acodec", "pcm_s16le",
//         "-ar", str(sr),
//         "-"
//     ]
#[cfg(feature = "ffmpeg-input")]
pub fn load_audio_waveform_with_ffmpeg(input_file: &str) -> Result<Vec<f32>, ffmpeg::Error> {
    ffmpeg::init()?;

    let mut ictx = ffmpeg::format::input(&input_file)?;
    let input_audio_stream = ictx
        .streams()
        .best(media::Type::Audio)
        .expect("could not find best audio stream");
    let audio_stream_index = input_audio_stream.index();
    // unsafe {
    //     println!(
    //         "input stream = {:?} par = {:?}",
    //         input,
    //         *input.parameters().as_ptr()
    //     );
    // }
    let context =
        ffmpeg::codec::context::Context::from_parameters(input_audio_stream.parameters())?;
    unsafe {
        //  Guessed Channel Layout: mono
        let par = input_audio_stream.parameters().as_mut_ptr();
        let x = (*par).ch_layout;

        // (*par).get
        // ffmpeg::ffi::av_c
        println!("channel = {:?} nb_channel {:?}", x.order, x.nb_channels);

        // if (*par).ch_layout. == 0 {
        //     (*par).ch_layout = ffmpeg::util::channel_layout::ChannelLayout::MONO.bits()
        // };
    }

    let mut decoder = context.decoder().audio()?;
    decoder.set_parameters(input_audio_stream.parameters())?;

    // let src_format = decoder.format();
    // let src_rate = decoder.rate();
    // let src_channel_layout = decoder.channel_layout();
    //

    let mut frame = frame::Audio::empty();
    let mut res = vec![];
    for (stream, packet) in ictx.packets() {
        if stream.index() != audio_stream_index {
            continue;
        }
        decoder.send_packet(&packet)?;
        while decoder.receive_frame(&mut frame).is_ok() {
            let mut out_frame = frame::Audio::empty();
            {
                let src_format = frame.format();
                // unsafe {
                //     let s = (*frame.as_ptr()).format;
                //     println!("frame format = {s:?}");
                // }
                let src_rate = frame.rate();
                let src_channel_layout = frame.channel_layout();

                let dst_rate = 16000u32;

                let mut swr = ffmpeg::software::resampling::Context::get(
                    src_format,
                    src_channel_layout,
                    src_rate,
                    ffmpeg::util::format::Sample::F32(ffmpeg::util::format::sample::Type::Packed), // AV_SAMPLE_FMT_FLT
                    ffmpeg::util::channel_layout::ChannelLayout::MONO,
                    dst_rate,
                )?;
                let _resample_res = swr.run(&frame, &mut out_frame)?;
            }
            // let in_format = frame.format();
            // unsafe {
            //     let raw_in = frame.as_mut_ptr();
            //     let raw_swr = swr.as_mut_ptr();
            //     // let res = av_channel_layout_compare(&(*raw_in).ch_layout, (*raw_swr).in_ch_layout);
            //     // println!("av_channel_layout_compare = {res}");
            // }
            unsafe {
                let out_frame = out_frame.as_mut_ptr();
                let tmp_slice = std::slice::from_raw_parts(
                    (*(*out_frame).extended_data) as *mut f32,
                    (*out_frame).nb_samples as usize,
                ); // the dst_format in swr is AV_SAMPLE_FMT_FLT, f32
                res.extend_from_slice(tmp_slice);
            }
        }
    }
    Ok(res)
}

#[cfg(feature = "hound-input")]
fn load_audio_waveform(filename: &str) -> hound::Result<(Vec<f32>, usize)> {
    let reader = hound::WavReader::open(filename)?;
    let spec = reader.spec();

    let _duration = reader.duration() as usize;
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate as usize;
    let _bits_per_sample = spec.bits_per_sample;
    let sample_format = spec.sample_format;

    assert_eq!(sample_rate, 16000, "The audio sample rate must be 16k.");
    assert_eq!(channels, 1, "The audio must be single-channel.");

    let max_int_val = 2_u32.pow(spec.bits_per_sample as u32 - 1) - 1;

    let floats = match sample_format {
        hound::SampleFormat::Float => reader.into_samples::<f32>().collect::<hound::Result<_>>()?,
        hound::SampleFormat::Int => reader
            .into_samples::<i32>()
            .map(|s| s.map(|s| s as f32 / max_int_val as f32))
            .collect::<hound::Result<_>>()?,
    };

    return Ok((floats, sample_rate));
}


use burn::prelude::Tensor;
use whisper::audio::{log_mel_spectrogram, N_SAMPLES, pad_or_trim};
use whisper::decoding::{DecodingOptions, UserSuppressTokens};
use crate::whsiper_helper::{WhichModel, WhisperHelper};

#[tokio::main]
async fn main() {
    let wav_file = "./audio16k.wav";

    cfg_if::cfg_if! {
        if #[cfg(feature = "wgpu-backend")] {
            type CurBackend = Wgpu<AutoGraphicsApi, f32, i32>;
            let device = WgpuDevice::BestAvailable;
        } else if #[cfg(feature = "torch-backend")] {
            type CurBackend = LibTorch<f32>;
            let device = LibTorchDevice::Cuda(0);
        }
    }
    println!("Loading waveform...");

    cfg_if::cfg_if! {
        if #[cfg(feature = "hound-input")] {
            let (waveform, _) = load_audio_waveform(wav_file).expect("");
        } else if #[cfg(feature = "ffmpeg-input")] {
            let waveform = load_audio_waveform_with_ffmpeg(wav_file).unwrap();
        }
    }
    println!("wave len = {}", waveform.len());
    // return;

    // let temperature = vec![];
    let _compression_ratio_threshold = Some(2.4_f32);
    let _logprob_threshold = Some(-1.0_f32);
    let _no_speech_threshold = Some(0.6_f32);
    let _clip_timestamps = vec![(0.0f32, 3.0f32)];

    let input_len = waveform.len();
    let audio = Tensor::<CurBackend, 2>::from_floats(
        burn::tensor::Data::new(waveform, [1, input_len].into()),
        &device,
    )
        .to_device(&device);

    let mut decode_options = DecodingOptions::default();
    decode_options.suppress_tokens = Some(UserSuppressTokens::Text("-1".to_string()));

    println!("======== loading model.........");
    let start_time = std::time::Instant::now();
    let helper: WhisperHelper<CurBackend> = WhisperHelper::new(WhichModel::Base, decode_options, &device).await;
    let loading_time = start_time.elapsed();
    println!(" cast {:?}", loading_time);
    println!("======== detecting language...");
    // Pad 30-seconds of silence to the input audio, for slicing
    let audio = pad_or_trim(&audio, N_SAMPLES);
    let mel = log_mel_spectrogram(audio);
    let detect_result = helper.detect_language(&mel);
    println!("res = {detect_result:#?}");
    println!("========= begin run............");
    let res = helper.run(mel);
    println!("run res = {res:#?}");
    return;
    // let temperature = vec![];
    // let compression_ratio_threshold = Some(2.4_f32);
    // let logprob_threshold = Some(-1.0_f32);
    // let no_speech_threshold = Some(0.6_f32);
    // let clip_timestamps = vec![(0.0f32, 3.0f32)];
    // let mut decode_options = whisper::decoding::DecodingOptions::default();
    //
    // let r = lib::transcribe(
    //     &helper.model,
    //     &helper.tokenizer,
    //     lang,
    //     waveform.clone(),
    //     temperature,
    //     compression_ratio_threshold,
    //     logprob_threshold,
    //     no_speech_threshold,
    //     clip_timestamps,
    //     &decode_options,
    //     &device,
    // );
    // return;
    // // let (text, _tokens) = match waveform_to_text(&whisper, &bpe, lang, waveform, 16000) {
    // //     Ok((text, tokens)) => (text, tokens),
    // //     Err(e) => {
    // //         eprintln!("Error during transcription: {}", e);
    // //         process::exit(1);
    // //     }
    // // };
    //
    // // fs::write(text_file, text).unwrap_or_else(|e| {
    // //     eprintln!("Error writing transcription file: {}", e);
    // //     process::exit(1);
    // // });
    //
    // // println!("Transcription finished.");
}


#[cfg(test)]
mod test {
    use num_traits::abs;
    use crate::{load_audio_waveform, load_audio_waveform_with_ffmpeg};


    #[test]
    fn test_input() {
        let input_file = "./audio16k.wav";
        let raw_input = "./audio16k.wav";

        let x = load_audio_waveform(input_file).unwrap();
        let y = load_audio_waveform_with_ffmpeg(raw_input).unwrap();
        assert_eq!(x.0.len(), y.len());
        for i in 0..x.0.len() {
            if abs(x.0[i] - y[i]) > 1e-5 {
                println!("{i} {} {} {}", x.0[i], y[i], abs(x.0[i] - y[i]));
            }
        }
        println!("len1 = {}, len2 = {}", x.0.len(), y.len());
        assert_eq!(x.0.len(), y.len());
    }
}