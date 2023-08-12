use super::models::WhichModel;
use clap::{command, Parser};

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
    #[arg(long, default_value = "tiny-en")]
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
