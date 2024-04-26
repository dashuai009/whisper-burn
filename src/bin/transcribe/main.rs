mod lib;
mod whsiper_helper;
use whisper::model::*;
use whisper::transcribe::waveform_to_text;

use strum::IntoEnumIterator;

use whisper::token::{Gpt2Tokenizer, Language};

use burn::record::{HalfPrecisionSettings, Recorder, RecorderError};

#[cfg(feature = "ffmpeg-input")]
use ffmpeg::{frame, media};
#[cfg(feature = "ffmpeg-input")]
use ffmpeg_next as ffmpeg;
use std::{clone, slice};

cfg_if::cfg_if! {
    if #[cfg(feature = "wgpu-backend")] {
        use burn_wgpu::{Wgpu, WgpuDevice, AutoGraphicsApi};
    } else if #[cfg(feature = "torch-backend")] {
        use burn_tch::{LibTorch, LibTorchDevice};
    }
}

use burn::{config::Config, module::Module, tensor::backend::Backend};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

use hound::{self, SampleFormat};

#[cfg(feature = "ffmpeg-input")]
fn load_audio_waveform_with_ffmpeg(input_file: &str) -> Result<Vec<f32>, ffmpeg::Error> {
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
        if (*par).channel_layout == 0 {
            (*par).channel_layout = ffmpeg::util::channel_layout::ChannelLayout::MONO.bits()
        };
    }

    let mut decoder = context.decoder().audio()?;
    decoder.set_parameters(input_audio_stream.parameters())?;

    let src_format = decoder.format();
    let src_rate = decoder.rate();
    let src_channel_layout = decoder.channel_layout();

    let dst_rate = 16000u32;
    let mut swr = ffmpeg::software::resampling::Context::get(
        src_format,
        src_channel_layout,
        src_rate,
        ffmpeg::util::format::Sample::F32(ffmpeg::util::format::sample::Type::Packed), // AV_SAMPLE_FMT_FLT
        ffmpeg::util::channel_layout::ChannelLayout::MONO,
        dst_rate,
    )?;

    let mut frame = frame::Audio::empty();
    let mut res = vec![];
    for (stream, packet) in ictx.packets() {
        if stream.index() != audio_stream_index {
            continue;
        }
        decoder.send_packet(&packet)?;
        while decoder.receive_frame(&mut frame).is_ok() {
            let mut out_frame = frame::Audio::empty();
            let _resample_res = swr.run(&frame, &mut out_frame)?;
            unsafe {
                let out_frame = out_frame.as_mut_ptr();
                let tmp_slice = slice::from_raw_parts(
                    (*(*out_frame).extended_data) as *mut f32,
                    (*out_frame).nb_samples as usize,
                ); // the dst_format in swr is AV_SAMPLE_FMT_FLT, f32
                res.extend_from_slice(tmp_slice);
            }
        }
    }
    Ok(res)
}
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
        SampleFormat::Float => reader.into_samples::<f32>().collect::<hound::Result<_>>()?,
        SampleFormat::Int => reader
            .into_samples::<i32>()
            .map(|s| s.map(|s| s as f32 / max_int_val as f32))
            .collect::<hound::Result<_>>()?,
    };

    return Ok((floats, sample_rate));
}

fn load_whisper_model_file<B: Backend>(
    config: &WhisperConfig,
    filename: &str,
    device: &B::Device,
) -> Result<Whisper<B>, RecorderError> {
    let full_filename = format!("{filename}.pt");
    let load_args = LoadArgs::new(full_filename.parse().unwrap())
        .with_debug_print()
        .with_key_remap("mlp.0", "mlp0")
        .with_key_remap("mlp.2", "mlp2")
        .with_top_level_key("model_state_dict");
    PyTorchFileRecorder::<HalfPrecisionSettings>::new()
        .load(load_args, device)
        .map(|record| config.init(device).load_record(record))
}

use std::{env, fs, process};

#[tokio::main]
async fn main() {
    cfg_if::cfg_if! {
        if #[cfg(feature = "wgpu-backend")] {
            type CurBackend = Wgpu<AutoGraphicsApi, f32, i32>;
            let device = WgpuDevice::BestAvailable;
        } else if #[cfg(feature = "torch-backend")] {
            type CurBackend = LibTorch<f32>;
            let device = LibTorchDevice::Cuda(0);
        }
    }

    let args: Vec<String> = env::args().collect();

    if args.len() < 5 {
        eprintln!(
            "Usage: {} <model name> <audio file> <lang> <transcription file>",
            args[0]
        );
        process::exit(1);
    }

    let wav_file = &args[2];
    let text_file = &args[4];

    let lang_str = &args[3];
    let lang = match Language::iter().find(|lang| lang.as_str() == lang_str) {
        Some(lang) => lang,
        None => {
            eprintln!("Invalid language abbreviation: {}", lang_str);
            process::exit(1);
        }
    };

    let model_name = &args[1];

    println!("Loading waveform...");
    // let waveform = match load_audio_waveform_with_ffmpeg(wav_file) {
    //     Ok(w) => w,
    //     Err(e) => {
    //         eprintln!("Failed to load audio file: {}", e);
    //         process::exit(1);
    //     }
    // };
    // println!(" y = {}", waveform.len());
    let waveform = x.0;

    // let bpe = match Gpt2Tokenizer::new() {
    //     Ok(bpe) => bpe,
    //     Err(e) => {
    //         eprintln!("Failed to load tokenizer: {}", e);
    //         process::exit(1);
    //     }
    // };

    // let whisper_config = match WhisperConfig::load(&format!("{}.cfg", model_name)) {
    //     Ok(config) => config,
    //     Err(e) => {
    //         eprintln!("Failed to load whisper config: {}", e);
    //         process::exit(1);
    //     }
    // };

    // println!("Loading model...");
    // let whisper = match load_whisper_model_file::<CurBackend>(&whisper_config, model_name, &device)
    // {
    //     Ok(whisper_model) => whisper_model,
    //     Err(e) => {
    //         eprintln!("Failed to load whisper model file: {}", e);
    //         process::exit(1);
    //     }
    // };
    let helper: whsiper_helper::WhisperHelper::<CurBackend> = whsiper_helper::WhisperHelper::new(whsiper_helper::WhichModel::Base, &device).await;

    let temperature = vec![];
    let compression_ratio_threshold = Some(2.4_f32);
    let logprob_threshold = Some(-1.0_f32);
    let no_speech_threshold = Some(0.6_f32);
    let clip_timestamps = vec![(0.0f32, 3.0f32)];
    let mut decode_options = whisper::decoding::DecodingOptions::default();

    let r = lib::transcribe(
        &helper.model,
        &helper.tokenizer,
        lang,
        waveform.clone(),
        temperature,
        compression_ratio_threshold,
        logprob_threshold,
        no_speech_threshold,
        clip_timestamps,
        &decode_options,
        &device,
    );
    return;
    // let (text, _tokens) = match waveform_to_text(&whisper, &bpe, lang, waveform, 16000) {
    //     Ok((text, tokens)) => (text, tokens),
    //     Err(e) => {
    //         eprintln!("Error during transcription: {}", e);
    //         process::exit(1);
    //     }
    // };

    // fs::write(text_file, text).unwrap_or_else(|e| {
    //     eprintln!("Error writing transcription file: {}", e);
    //     process::exit(1);
    // });

    // println!("Transcription finished.");
}



#[cfg(test)]
mod test{
    use num_traits::abs;
    use crate::{load_audio_waveform, load_audio_waveform_with_ffmpeg};


    #[test]
    fn test_input(){
        let input_file = "./audio16k.wav";

        let x = load_audio_waveform(input_file).unwrap();
        let y = load_audio_waveform_with_ffmpeg(input_file).unwrap();
        for i in 0..x.0.len(){
            if abs(x.0[i]- y[i]) > 2e-5 {
                println!("{i} {} {} {}",x.0[i], y[i], abs(x.0[i]- y[i]));
            }
        }
        println!("len1 = {}, len2 = {}", x.0.len(), y.len());
        assert_eq!(x.0.len(), y.len());
    }
}