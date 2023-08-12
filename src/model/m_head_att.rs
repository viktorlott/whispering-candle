use candle::{Result, Tensor};
use candle_nn::{ops::softmax, VarBuilder};

use super::{linear, linear_no_bias, Linear};

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L62
pub struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
    span: tracing::Span,
}

impl MultiHeadAttention {
    pub fn load(n_state: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "multi-head-attn");
        let query = linear(n_state, n_state, vb.pp("q_proj"))?;
        let value = linear(n_state, n_state, vb.pp("v_proj"))?;
        let key = linear_no_bias(n_state, n_state, vb.pp("k_proj"))?;
        let out = linear(n_state, n_state, vb.pp("out_proj"))?;
        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
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
        let q = self.query.forward(x)?;
        let k = self.key.forward(xa.unwrap_or(x))?;
        let v = self.value.forward(xa.unwrap_or(x))?;
        let wv = self.qkv_attention(&q, &k, &v, mask)?;
        let out = self.out.forward(&wv)?;
        Ok(out)
    }

    fn reshape_head(&self, x: &Tensor) -> Result<Tensor> {
        let (n_batch, n_ctx, n_state) = x.dims3()?;
        let target_dims = &[n_batch, n_ctx, self.n_head, n_state / self.n_head];
        x.reshape(target_dims)?.transpose(1, 2)
    }

    fn qkv_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_, n_ctx, n_state) = q.dims3()?;
        let scale = ((n_state / self.n_head) as f64).powf(-0.25);
        let q = (self.reshape_head(q)? * scale)?;
        let k = (self.reshape_head(k)?.transpose(2, 3)? * scale)?;
        let v = self.reshape_head(v)?.contiguous()?;
        let mut qk = q.matmul(&k)?;
        if let Some(mask) = mask {
            let mask = mask.narrow(0, 0, n_ctx)?.narrow(1, 0, n_ctx)?;
            qk = qk.broadcast_add(&mask)?
        }
        let w = softmax(&qk, candle::D::Minus1)?;
        let wv = w.matmul(&v)?.transpose(1, 2)?.flatten_from(2)?;
        Ok(wv)
    }
}
