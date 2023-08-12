use candle::{Result, Tensor};
use candle_nn::VarBuilder;

use super::{audio_enc::AudioEncoder, text_dec::TextDecoder, Config};

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L221
pub struct Whisper {
    pub encoder: AudioEncoder,
    pub decoder: TextDecoder,
    pub config: Config,
}

impl Whisper {
    pub fn load(vb: &VarBuilder, config: Config) -> Result<Self> {
        let encoder = AudioEncoder::load(vb.pp("model.encoder"), &config)?;
        let decoder = TextDecoder::load(vb.pp("model.decoder"), &config)?;
        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    pub fn is_multilingual(&self) -> bool {
        self.config.vocab_size == 51865
    }

    #[allow(dead_code)]
    pub fn forward(&self, mel: &Tensor, tokens: &Tensor) -> Result<Tensor> {
        let enc = self.encoder.forward(mel)?;
        let dec = self.decoder.forward(tokens, &enc)?;
        Ok(dec)
    }
}
