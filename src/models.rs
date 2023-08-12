use clap::ValueEnum;
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
