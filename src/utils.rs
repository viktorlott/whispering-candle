use anyhow::{Error as E, Result};
use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use std::fs::read_to_string;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

use super::args::Args;
use super::decoder::Decoder;
use super::model::Whisper;
use super::multilingual;
use super::{audio, constants};

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else {
        let device = Device::cuda_if_available(0)?;
        if !device.is_cuda() {
            // Running on CPU, to run on GPU, build this example with `--features cuda`
            println!("Running on CPU");
        }
        Ok(device)
    }
}

pub fn setup_tracing(args: &Args) -> Option<tracing_chrome::FlushGuard> {
    // Importing required modules and traits
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    if args.tracing {
        println!("tracing...");
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    }
}

/// Extracting model information
pub fn get_model_info(args: &Args) -> (&'static str, &'static str) {
    let (default_model, default_revision) = args.model.into();
    (default_model, default_revision)
}

// Model data setup: Either use local or download from remote
pub fn setup_model_data(
    args: &Args,
    default_model: &str,
    default_revision: &str,
) -> Result<(PathBuf, PathBuf, PathBuf, PathBuf)> {
    let path = std::path::PathBuf::from(default_model.clone());

    let model_id = args
        .model_id
        .as_ref()
        .map(AsRef::as_ref)
        .unwrap_or(default_model);

    let revision = args
        .revision
        .as_ref()
        .map(AsRef::as_ref)
        .unwrap_or(default_revision);

    if path.exists() {
        get_local_model_data(path, args)
    } else {
        download_model_data(&model_id, &revision, args)
    }
}

fn get_local_model_data(
    path: PathBuf,
    args: &Args,
) -> Result<(PathBuf, PathBuf, PathBuf, PathBuf)> {
    let mut config_filename = path.clone();
    config_filename.push("config.json");

    let mut tokenizer_filename = path.clone();
    tokenizer_filename.push("tokenizer.json");

    let mut model_filename = path;
    model_filename.push("model.safetensors");

    let input_filename = PathBuf::from(args.input.as_ref().expect("Expected input argument"));

    Ok((
        config_filename,
        tokenizer_filename,
        model_filename,
        input_filename,
    ))
}

fn download_model_data(
    model_id: &str,
    revision: &str,
    args: &Args,
) -> Result<(PathBuf, PathBuf, PathBuf, PathBuf)> {
    let api = Api::new()?;
    let dataset = api.dataset("Narsil/candle-examples".to_string());
    let repo = api.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));

    let sample = if let Some(input) = &args.input {
        if let Some(sample) = input.strip_prefix("sample:") {
            dataset.get(&format!("samples_{sample}.wav"))?
        } else {
            PathBuf::from(input)
        }
    } else {
        // submitted: Downloading https://huggingface.co/datasets/Narsil/candle_demo/blob/main/samples_jfk.wav"
        // println!("No audio file");
        dataset.get("samples_jfk.wav")?
    };

    Ok((
        repo.get("config.json")?,
        repo.get("tokenizer.json")?,
        repo.get("model.safetensors")?,
        sample,
    ))
}

// Overview:
// * `PCM` - Pulse Code Modulation is often the starting point in digital audio processing. When
//   audio is recorded using a microphone and stored digitally, the analog waveform is typically
//   converted to digital using PCM. This method involves sampling the continuous audio waveform at
//   regular intervals and then quantizing each sample's amplitude to represent it in binary format.
//   The outcome is a sequence of digital values representing the audio signal.
//
// * `Feature Extraction - Mel` - For many audio processing tasks, especially in speech recognition,
//   it's beneficial to transform the raw PCM audio data into a different representation that
//   emphasizes certain features of the audio, discarding less crucial information.
//
// Steps:
// 1. `From PCM to Spectrogram` - Before extracting MFCCs, the PCM audio undergoes a transformation
//    into a spectrogram using the Short-Time Fourier Transform (STFT). This reveals how the power of
//    different frequencies in the audio signal varies over time.
//
// 2. `Spectrogram to Mel Spectrogram` - The frequency scale of the spectrogram is then adapted to
//    the Mel scale. This perceptual scale mirrors the human ear's response to varying frequencies,
//    resulting in a "Mel spectrogram."
//
// 3. `Mel Spectrogram to MFCCs` - The Mel spectrogram is further transformed to extract MFCCs using
//    the Discrete Cosine Transform (DCT). MFCCs capture the spectral characteristics of the audio,
//    reducing sensitivity to variations that may not be perceptually significant.
pub fn process_audio(input_path: &Path, device: &Device) -> Result<Tensor> {
    let mel_bytes = include_bytes!("assets/melfilters.bytes");
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];

    <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);

    let mut input = std::fs::File::open(input_path)?;
    let (header, data) = wav::read(&mut input)?;

    if header.sampling_rate != constants::SAMPLE_RATE as u32 {
        anyhow::bail!(
            "wav file must have a {} sampling rate",
            constants::SAMPLE_RATE
        )
    }

    let data = data.as_sixteen().expect("Expected 16 bit wav file");
    let pcm_data: Vec<_> = data.iter().map(|v| *v as f32 / 32768.).collect();

    let mel = audio::pcm_to_mel(&pcm_data, &mel_filters)?;
    let mel_len = mel.len();

    let mel = Tensor::from_vec(
        mel,
        (1, constants::N_MELS, mel_len / constants::N_MELS),
        device,
    )?;

    Ok(mel)
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

pub fn setup_and_run_model(
    vb: VarBuilder,
    config_filename: &Path,
    tokenizer_filename: &Path,
    args: &Args,
    mel: &Tensor,
    device: &Device,
) -> Result<()> {
    let config = serde_json::from_str(&read_to_string(config_filename)?)?;
    let model = Whisper::load(&vb, config)?;

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let language_token = if model.is_multilingual() {
        Some(multilingual::detect_language(&model, &tokenizer, &mel)?)
    } else {
        None
    };

    let mut dc = Decoder::new(model, tokenizer, args.seed, &device, language_token)?;
    dc.run(&mel)?;

    Ok(())
}

pub const fn factorial_sequence<const N: usize>(positive: bool) -> [f32; N] {
    let mut list: [f32; N] = [0.0; N];
    let mut index = 1;
    let mut factorial: usize = 1;

    loop {
        if index > N {
            break;
        }

        factorial *= (if positive {
            2 * index + 1
        } else {
            2 * index - 1
        }) * (2 * index);
        list[index - 1] = factorial as f32;

        index += 1;
    }

    list
}
