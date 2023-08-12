use candle::{Result, Tensor};
use candle_nn::{Embedding, LayerNorm, VarBuilder};

use super::{embedding, layer_norm, res_att::ResidualAttentionBlock, Config};

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L176
pub struct TextDecoder {
    token_embedding: Embedding,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln: LayerNorm,
    mask: Tensor,
    span: tracing::Span,
}

impl TextDecoder {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "text-decoder");
        let n_state = cfg.d_model;
        let n_head = cfg.decoder_attention_heads;
        let n_ctx = cfg.max_target_positions;
        let token_embedding = embedding(cfg.vocab_size, n_state, vb.pp("embed_tokens"))?;
        let positional_embedding = vb.get((n_ctx, n_state), "embed_positions.weight")?;
        let blocks = (0..cfg.decoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(n_state, n_head, true, vb.pp(&format!("layers.{i}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let ln = layer_norm(n_state, vb.pp("layer_norm"))?;
        let mask: Vec<_> = (0..n_ctx)
            .flat_map(|i| (0..n_ctx).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        let mask = Tensor::from_vec(mask, (n_ctx, n_ctx), vb.device())?;

        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
            span,
        })
    }

    pub fn forward(&self, x: &Tensor, xa: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x_dims = x.dims();
        let last = x_dims[x_dims.len() - 1];
        let token_embedding = self.token_embedding.forward(x)?;
        let positional_embedding = self.positional_embedding.narrow(0, 0, last)?;
        let mut x = token_embedding.broadcast_add(&positional_embedding)?;
        for block in self.blocks.iter() {
            x = block.forward(&x, Some(xa), Some(&self.mask))?;
        }
        let x = self.ln.forward(&x)?;
        let w = self
            .token_embedding
            .embeddings()
            .broadcast_left(x_dims[0])?;
        let logits = x.matmul(&w.t()?)?;
        Ok(logits)
    }
}
