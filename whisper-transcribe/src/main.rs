#[cfg(feature = "ffmpeg-input")]
extern crate ffmpeg_next;


#[cfg(feature = "ffmpeg-input")]
use ffmpeg::{frame, media};
#[cfg(feature = "ffmpeg-input")]
use ffmpeg_next as ffmpeg;

use cfg_if;



cfg_if::cfg_if! {
    if #[cfg(feature = "wgpu-backend")] {
        use burn_wgpu::{Wgpu, WgpuDevice, AutoGraphicsApi};
    } else if #[cfg(feature = "torch-backend")] {
        use burn_tch::{LibTorch, LibTorchDevice};
    }
}

/// audio to 16k Hz
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
pub fn load_audio_waveform(input_file: &str) -> Result<Vec<f32>, ffmpeg::Error> {
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
    let mut context =
        ffmpeg::codec::context::Context::from_parameters(input_audio_stream.parameters())?;
    unsafe {
        //  Guessed Channel Layout: mono
        let xx = context.as_mut_ptr();
        if (*xx).ch_layout.order == ffmpeg_next::ffi::AVChannelOrder::AV_CHANNEL_ORDER_UNSPEC {
            // let mut NewLayout = std::mem::zeroed::<ffmpeg_next::ffi::AVChannelLayout>();
            // ffmpeg_next::ffi::av_channel_layout_default(&mut NewLayout, (*xx).ch_layout.nb_channels);
            // let res = ffmpeg_next::ffi::av_channel_layout_copy(&mut (*xx).ch_layout, &NewLayout);
            // println!("res = {res}");
            // std::mem::forget(NewLayout);
        }
        // let par = input_audio_stream.parameters().as_mut_ptr();
        // let x = (*par).ch_layout;

        // (*par).get
        // ffmpeg::ffi::av_c
        // println!("channel = {:?} nb_channel {:?}", x.order, x.nb_channels);

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

#[cfg(not(feature = "ffmpeg-input"))]
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


use whisper::decoding::{DecodingOptions, UserSuppressTokens};
use whisper::whisper_helper::{WhichModel, WhisperHelper};

#[cfg(feature = "wgpu-backend")]
fn init_model() -> (WhisperHelper<Wgpu<AutoGraphicsApi, f32, i32>>, WgpuDevice) {
    let device = WgpuDevice::BestAvailable;
    let helper = WhisperHelper::new(WhichModel::Tiny, &device);
    return (helper, device);
}

#[cfg(not(feature = "wgpu-backend"))]
fn init_model() -> (WhisperHelper<LibTorch<f32>>, LibTorchDevice) {
    let device = LibTorchDevice::Cuda(0);
    let helper: WhisperHelper<CurBackend> = WhisperHelper::new(WhichModel::Medium, &device);
    return (helper, device);
}

fn main() {
    let fake_endpoint = "https://hf-mirror.com".to_string();
    std::env::set_var("HF_ENDPOINT", &fake_endpoint);
    let wav_files = [
        // "./tmp/20.mp4",
        "./tmp/40.mp4",
        // "./tmp/70.mp4",
        // "./tmp/300.mp4",
        // "./tmp/600.mp4"
    ];
    // let wav_file = "./audio.wav";

    println!("======== loading model.........");
    let start_time = std::time::Instant::now();
    let (helper, device) = init_model();
    let loading_time = start_time.elapsed();
    println!("loading model cast {:?}", loading_time);

    println!("Loading waveform...");
    for file in wav_files {
        let waveform = load_audio_waveform(file).expect("");
        println!("wave len = {}", waveform.len());

        let mut decode_options = DecodingOptions::default();
        decode_options.suppress_tokens = Some(UserSuppressTokens::Text("-1".to_string()));

        println!("========= begin run............");
        let now = std::time::Instant::now();
        let mut res= helper.run(&waveform, 8, decode_options, &device);
        for i in &mut res{
            i.tokens= vec![];
        }
        println!("run res = {res:#?}\ndecoding cost: {:?}", now.elapsed());
    }
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