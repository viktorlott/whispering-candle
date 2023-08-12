use clap::{command, Parser, ValueEnum};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    pub cpu: bool,

    #[arg(long)]
    pub model_id: Option<String>,

    /// The model to use, check out available models:
    /// https://huggingface.co/models?search=whisper
    #[arg(long)]
    pub revision: Option<String>,

    /// The model to be used, can be tiny, small, medium.
    #[arg(long, default_value = "tiny")]
    pub model: WhichModel,

    /// The input to be processed, in wav format, will default to `jfk.wav`. Alternatively
    /// this can be set to sample:jfk, sample:gb1, ... to fetch a sample from the following
    /// repo: https://huggingface.co/datasets/Narsil/candle_demo/
    #[arg(long)]
    pub input: Option<String>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    pub seed: u64,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    pub tracing: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
#[penum::into((&'static str, &'static str))]
pub enum WhichModel {
    // Supported safetensors (https://huggingface.co/docs/safetensors/index)
    // Some of these are not yet merged, hence the pr refs
    Tiny = ("openai/whisper-tiny", "main"),
    TinyEn = ("openai/whisper-tiny.en", "refs/pr/15"),
    Base = ("openai/whisper-base", "refs/pr/22"),
    BaseEn = ("openai/whisper-base.en", "refs/pr/13"),
    SmallEn = ("openai/whisper-small.en", "refs/pr/10"),
    MediumEn = ("openai/whisper-medium.en", "refs/pr/11"),
    LargeV2 = ("openai/whisper-large-v2", "refs/pr/57"),
}
