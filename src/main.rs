mod args;
mod audio;
mod constants;
mod decoder;
mod model;
mod models;
mod multilingual;
mod utils;

use anyhow::Result;
use candle_nn::VarBuilder;
use clap::Parser;

use args::Args;
use utils::{
    device, get_model_info, process_audio, setup_and_run_model, setup_model_data, setup_tracing,
};

// Resources:
// - https://medium.com/@bofenghuang7/what-i-learned-from-whisper-fine-tuning-event-2a68dab1862
// - https://github.com/openai/whisper/blob/main/whisper/model.py
fn main() -> Result<()> {
    // Parse command line arguments.
    let args = Args::parse();

    // Set up tracing based on the provided command line arguments.
    let _guard = setup_tracing(&args);

    // Create a computation device (e.g., CPU or GPU) based on the arguments.
    let device = device(args.cpu)?;

    // Extract the default model name and its revision.
    let (default_model, default_revision) = get_model_info(&args);

    // Set up the required paths for model data, either by locating locally or downloading.
    let (config_filename, tokenizer_filename, weights_filename, input_path) =
        setup_model_data(&args, &default_model, &default_revision)?;

    // Process the audio data, converting it into a format suitable for the model.
    let mel = process_audio(&input_path, &device)?;

    // Load model weights from the specified file.
    let weights = unsafe { candle::safetensors::MmapedFile::new(weights_filename)? };
    let weights = weights.deserialize()?;

    // Build a variable from the deserialized weights.
    let vb = VarBuilder::from_safetensors(vec![weights], constants::DTYPE, &device);

    // Set up the model and execute it on the processed audio data.
    setup_and_run_model(
        vb,
        &config_filename,
        &tokenizer_filename,
        &args,
        &mel,
        &device,
    )?;

    Ok(())
}
