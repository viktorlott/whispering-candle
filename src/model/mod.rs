mod audio_enc;
mod m_head_att;
mod res_att;
mod text_dec;
mod whisper;

use candle::{Device, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Embedding, LayerNorm, VarBuilder};
use serde::Deserialize;

pub use whisper::Whisper;

// The names in comments correspond to the original implementation:
// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L17
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub num_mel_bins: usize,            // n_mels
    pub max_source_positions: usize,    // n_audio_ctx
    pub d_model: usize,                 // n_audio_state
    pub encoder_attention_heads: usize, // n_audio_head
    pub encoder_layers: usize,          // n_audio_layer
    pub vocab_size: usize,              // n_vocab
    pub max_target_positions: usize,    //  n_text_ctx
    // pub n_text_state: usize,
    pub decoder_attention_heads: usize, // n_text_head
    pub decoder_layers: usize,          // n_text_layer
    pub suppress_tokens: Vec<u32>,
}

impl Config {
    #[allow(dead_code)]
    pub fn tiny_en() -> Self {
        // List of tokens to be suppressed during the decoding process.
        // Extracted from the Whisper project's _get_suppress_tokens function.
        // These tokens might represent non-speech elements, irrelevant characters, etc.
        // From the _get_suppress_tokens function + 50362 (no timestamp)
        // https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/decoding.py#L605
        let suppress_tokens = vec![
            1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93,
            357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377, 1391,
            1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600, 4808, 5299, 5855, 6329,
            7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306,
            16410, 16791, 17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409,
            34949, 40283, 40493, 40549, 47282, 49146, 50257, 50357, 50358, 50359, 50360, 50361,
            50362,
        ];

        Self {
            num_mel_bins: 80,
            vocab_size: 51864,
            max_source_positions: 1500,
            d_model: 384,
            encoder_attention_heads: 6,
            encoder_layers: 4,
            max_target_positions: 448,
            // n_text_state: 384,
            decoder_attention_heads: 6,
            decoder_layers: 4,
            suppress_tokens,
        }
    }
}

fn embedding(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((vocab_size, hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, hidden_size))
}
//
// We wrap the `Linear` layer here to add some tracing so that it's easier to profile the resulting
// model.
#[derive(Debug)]
pub struct Linear {
    inner: candle_nn::Linear,
    span: tracing::Span,
}

impl Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

fn linear(size1: usize, size2: usize, vb: VarBuilder) -> Result<Linear> {
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    let inner = candle_nn::linear(size1, size2, vb)?;
    Ok(Linear { inner, span })
}

fn linear_no_bias(size1: usize, size2: usize, vb: VarBuilder) -> Result<Linear> {
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    let inner = candle_nn::linear_no_bias(size1, size2, vb)?;
    Ok(Linear { inner, span })
}

fn conv1d(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    config: Conv1dConfig,
    vb: VarBuilder,
) -> Result<Conv1d> {
    let weight = vb.get((out_channels, in_channels, kernel_size), "weight")?;
    let bias = vb.get(out_channels, "bias")?;
    Ok(Conv1d::new(weight, Some(bias), config))
}

fn layer_norm(size: usize, vb: VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get(size, "weight")?;
    let bias = vb.get(size, "bias")?;
    Ok(LayerNorm::new(weight, bias, 1e-5))
}

fn sinusoids(length: usize, channels: usize) -> Result<Tensor> {
    let max_timescale = 10000f32;
    let log_timescale_increment = max_timescale.ln() / (channels / 2 - 1) as f32;
    let inv_timescales: Vec<_> = (0..channels / 2)
        .map(|i| (i as f32 * (-log_timescale_increment)).exp())
        .collect();
    let inv_timescales = Tensor::new(inv_timescales.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
    let arange = Tensor::arange(0, length as u32, &Device::Cpu)?
        .to_dtype(candle::DType::F32)?
        .unsqueeze(1)?;
    let sh = (length, channels / 2);
    let scaled_time = (arange.broadcast_as(sh)? * inv_timescales.broadcast_as(sh)?)?;
    let sincos = Tensor::cat(&[scaled_time.sin()?, scaled_time.cos()?], 1)?;
    Ok(sincos)
}
