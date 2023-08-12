mod audio_enc;
mod config;
mod m_head_att;
mod res_att;
mod text_dec;
mod whisper;

use candle::{Device, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Embedding, LayerNorm, VarBuilder};

pub use config::Config;
pub use whisper::Whisper;

//
// We wrap the `Linear` layer here to add some tracing so that it's easier to profile the resulting
// model.
#[derive(Debug)]
struct Linear {
    inner: candle_nn::Linear,
    span: tracing::Span,
}

impl Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

fn embedding(vocab_size: usize, hidden_size: usize, vb: VarBuilder) -> Result<Embedding> {
    let embeddings = vb.get((vocab_size, hidden_size), "weight")?;
    Ok(Embedding::new(embeddings, hidden_size))
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
