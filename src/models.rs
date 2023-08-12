use clap::ValueEnum;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum WhichModel {
    Tiny,
    TinyEn,
    SmallEn,
    MediumEn,
}

impl WhichModel {
    pub fn model_and_revision(&self) -> (&'static str, &'static str) {
        match self {
            Self::Tiny => ("openai/whisper-tiny", "main"),
            Self::TinyEn => ("openai/whisper-tiny.en", "refs/pr/15"),
            Self::SmallEn => ("openai/whisper-small.en", "refs/pr/10"),
            Self::MediumEn => ("openai/whisper-medium.en", "refs/pr/11"),
        }
    }
}
