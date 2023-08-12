use candle::{Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, VarBuilder};

use super::{conv1d, layer_norm, res_att::ResidualAttentionBlock, sinusoids, Config};

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L143
pub struct AudioEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
    span: tracing::Span,
    conv1_span: tracing::Span,
    conv2_span: tracing::Span,
}

impl AudioEncoder {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "audio-encoder");
        let conv1_span = tracing::span!(tracing::Level::TRACE, "conv1");
        let conv2_span = tracing::span!(tracing::Level::TRACE, "conv2");
        let n_state = cfg.d_model;
        let n_head = cfg.encoder_attention_heads;
        let n_ctx = cfg.max_source_positions;
        let cfg1 = Conv1dConfig {
            padding: 1,
            stride: 1,
        };
        let cfg2 = Conv1dConfig {
            padding: 1,
            stride: 2,
        };
        let conv1 = conv1d(cfg.num_mel_bins, n_state, 3, cfg1, vb.pp("conv1"))?;
        let conv2 = conv1d(n_state, n_state, 3, cfg2, vb.pp("conv2"))?;
        let positional_embedding = sinusoids(n_ctx, n_state)?.to_device(vb.device())?;
        let blocks = (0..cfg.encoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(n_state, n_head, false, vb.pp(&format!("layers.{i}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let ln_post = layer_norm(n_state, vb.pp("layer_norm"))?;
        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            blocks,
            ln_post,
            conv1_span,
            conv2_span,
            span,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = {
            let _enter = self.conv1_span.enter();
            self.conv1.forward(x)?.gelu()?
        };
        let x = {
            let _enter = self.conv2_span.enter();
            self.conv2.forward(&x)?.gelu()?
        };
        let x = x.transpose(1, 2)?;
        let (_bsize, seq_len, _hidden) = x.dims3()?;
        let positional_embedding = self.positional_embedding.narrow(0, 0, seq_len)?;
        let mut x = x.broadcast_add(&positional_embedding)?;
        for block in self.blocks.iter() {
            x = block.forward(&x, None, None)?
        }
        let x = self.ln_post.forward(&x)?;
        Ok(x)
    }
}
