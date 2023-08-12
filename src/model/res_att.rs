use candle::{Result, Tensor};
use candle_nn::{LayerNorm, VarBuilder};

use super::{layer_norm, linear, m_head_att::MultiHeadAttention, Linear};

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L111
pub struct ResidualAttentionBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    cross_attn: Option<(MultiHeadAttention, LayerNorm)>,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    mlp_ln: LayerNorm,
    span: tracing::Span,
}

impl ResidualAttentionBlock {
    pub fn load(n_state: usize, n_head: usize, ca: bool, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "residual-attn");
        let attn = MultiHeadAttention::load(n_state, n_head, vb.pp("self_attn"))?;
        let attn_ln = layer_norm(n_state, vb.pp("self_attn_layer_norm"))?;
        let cross_attn = if ca {
            let cross_attn = MultiHeadAttention::load(n_state, n_head, vb.pp("encoder_attn"))?;
            let cross_attn_ln = layer_norm(n_state, vb.pp("encoder_attn_layer_norm"))?;
            Some((cross_attn, cross_attn_ln))
        } else {
            None
        };
        let n_mlp = n_state * 4;
        let mlp_linear1 = linear(n_state, n_mlp, vb.pp("fc1"))?;
        let mlp_linear2 = linear(n_mlp, n_state, vb.pp("fc2"))?;
        let mlp_ln = layer_norm(n_state, vb.pp("final_layer_norm"))?;
        Ok(Self {
            attn,
            attn_ln,
            cross_attn,
            mlp_linear1,
            mlp_linear2,
            mlp_ln,
            span,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let attn = self.attn.forward(&self.attn_ln.forward(x)?, None, mask)?;
        let mut x = (x + attn)?;
        if let Some((attn, ln)) = &self.cross_attn {
            x = (&x + attn.forward(&ln.forward(&x)?, xa, None)?)?;
        }
        let mlp = self.mlp_linear2.forward(
            &self
                .mlp_linear1
                .forward(&self.mlp_ln.forward(&x)?)?
                .gelu()?,
        )?;

        x + mlp
    }
}
