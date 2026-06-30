"""
Attention Is All You Need — Pure NumPy with Manual Backprop + Training
arXiv:1706.03762v7

Trains a Transformer on sequence copying from scratch. Each layer
implements its own forward() and backward() methods.
"""
import math
import time
import os
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  PARAMETER & LAYER BASE
# ═══════════════════════════════════════════════════════════════════════

class Param:
    """Trainable parameter with data and gradient."""
    def __init__(self, *shape, scale=0.02):
        self.data = np.random.randn(*shape).astype(np.float64) * scale
        self.grad = np.zeros_like(self.data)

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad.fill(0)


class Layer:
    """Base layer. Subclasses implement forward() and backward()."""

    def params(self):
        """Yield all Param objects recursively (no duplicates by id)."""
        seen = set()
        for attr in self.__dict__.values():
            if isinstance(attr, Param):
                if id(attr) not in seen:
                    seen.add(id(attr))
                    yield attr
            elif isinstance(attr, Layer):
                for p in attr.params():
                    if id(p) not in seen:
                        seen.add(id(p))
                        yield p
            elif isinstance(attr, list):
                for item in attr:
                    if isinstance(item, Layer):
                        for p in item.params():
                            if id(p) not in seen:
                                seen.add(id(p))
                                yield p

    def param_count(self):
        return sum(p.data.size for p in self.params())

    def zero_grad(self):
        for p in self.params():
            p.zero_grad()


# ═══════════════════════════════════════════════════════════════════════
#  BASIC LAYERS
# ═══════════════════════════════════════════════════════════════════════

class Linear(Layer):
    """y = x @ W + b. Supports external shared weight with transpose for proj."""
    def __init__(self, d_in, d_out, weight=None, transpose_w=False):
        self._transpose_w = transpose_w
        if weight is not None:
            self.w = weight  # external shared Param
        else:
            limit = math.sqrt(6 / (d_in + d_out))
            self.w = Param(d_in, d_out)
            self.w.data = np.random.uniform(-limit, limit, (d_in, d_out)).astype(np.float64)
        self.b = Param(d_out)
        self.b.data.fill(0.0)
        self._x = None  # cache for backward

    def _get_w(self):
        """Return the effective weight matrix for forward."""
        return self.w.data.T if self._transpose_w else self.w.data

    def forward(self, x):
        self._x = x
        return x @ self._get_w() + self.b.data

    def backward(self, dout):
        """dout: (..., d_out). Compute gradients for w, b, and dx."""
        x = self._x
        orig_shape = x.shape
        if x.ndim > 2:
            x_2d = x.reshape(-1, x.shape[-1])
            dout_2d = dout.reshape(-1, dout.shape[-1])
        else:
            x_2d = x
            dout_2d = dout

        if self._transpose_w:
            # y = x @ W^T + b   where W is (vocab, d_model)
            # dL/dW[q,r] = sum_p dout[p,q] * x[p,r] = (dout^T @ x)[q,r]
            self.w.grad += dout_2d.T @ x_2d   # (vocab, d_model)
            dLdx = dout_2d @ self.w.data       # (N, d_model)
        else:
            # y = x @ W + b   where W is (d_in, d_out)
            self.w.grad += x_2d.T @ dout_2d
            dLdx = dout_2d @ self.w.data.T

        self.b.grad += dout_2d.sum(axis=0)
        return dLdx.reshape(orig_shape)


class LayerNorm(Layer):
    """y = gamma * (x - mean) / sqrt(var + eps) + beta"""
    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.gamma = Param(dim)
        self.gamma.data[:] = 1.0
        self.beta = Param(dim)
        self.beta.data[:] = 0.0
        self._cache = None

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        self._cache = (x, mean, var, x_norm)
        return x_norm * self.gamma.data + self.beta.data

    def backward(self, dout):
        x, mean, var, x_norm = self._cache
        N = x.shape[-1]
        std = np.sqrt(var + self.eps)

        # dL/dgamma = sum(dout * x_norm, axis=0)
        self.gamma.grad += (dout * x_norm).sum(axis=tuple(range(dout.ndim - 1)))
        # dL/dbeta = sum(dout, axis=0)
        self.beta.grad += dout.sum(axis=tuple(range(dout.ndim - 1)))

        # dL/dx_norm = dout * gamma
        dx_norm = dout * self.gamma.data

        # dL/dx = (1/N) * std^{-1} * (N * dx_norm - sum(dx_norm) - x_norm * sum(dx_norm * x_norm))
        sum_dx = dx_norm.sum(axis=-1, keepdims=True)
        sum_dx_xn = (dx_norm * x_norm).sum(axis=-1, keepdims=True)
        dx = (dx_norm - sum_dx / N - x_norm * sum_dx_xn / N) / std
        return dx


class Dropout(Layer):
    """Section 5.4: dropout applied during training, scaled by 1/(1-rate)."""
    def __init__(self, rate=0.1):
        self.rate = rate
        self._mask = None

    def forward(self, x, training=True):
        if not training or self.rate == 0:
            return x
        self._mask = np.random.binomial(1, 1 - self.rate, x.shape).astype(np.float64)
        return x * self._mask / (1 - self.rate)

    def backward(self, dout):
        if self._mask is None:
            return dout
        return dout * self._mask / (1 - self.rate)


class Embedding(Layer):
    """Lookup table. Forward: indices -> vectors. Backward: accumulate to correct rows."""
    def __init__(self, vocab, dim, weight=None):
        self.w = weight if weight is not None else Param(vocab, dim)
        self._indices = None

    def forward(self, indices):
        self._indices = indices
        return self.w.data[indices]

    def backward(self, dout):
        """Accumulate gradients at the positions used in forward."""
        for i in range(self._indices.shape[0]):
            for j in range(self._indices.shape[1]):
                idx = self._indices[i, j]
                self.w.grad[idx] += dout[i, j]
        return None  # no gradient to input indices


# ═══════════════════════════════════════════════════════════════════════
#  ATTENTION
# ═══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(Layer):
    """Section 3.2.2 with optional attention dropout (Section 5.4)."""
    def __init__(self, d_model=512, h=8, dropout=0.1):
        assert d_model % h == 0
        self.d_model = d_model
        self.h = h
        self.dk = d_model // h
        self.wq = Linear(d_model, d_model)
        self.wk = Linear(d_model, d_model)
        self.wv = Linear(d_model, d_model)
        self.wo = Linear(d_model, d_model)
        self.attn_dropout = Dropout(dropout)
        self._cache = None

    def forward(self, q_in, k_in, v_in, mask=None, training=True):
        B = q_in.shape[0]
        L, S = q_in.shape[1], k_in.shape[1]

        # Project
        Q = self.wq.forward(q_in).reshape(B, L, self.h, self.dk).transpose(0, 2, 1, 3)
        K = self.wk.forward(k_in).reshape(B, S, self.h, self.dk).transpose(0, 2, 1, 3)
        V = self.wv.forward(v_in).reshape(B, S, self.h, self.dk).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = Q @ K.transpose(0, 1, 3, 2) / math.sqrt(self.dk)
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        attn_softmax = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn_softmax = attn_softmax / attn_softmax.sum(axis=-1, keepdims=True)

        # Attention dropout (Section 5.4)
        attn = self.attn_dropout.forward(attn_softmax, training=training)

        out = attn @ V  # (B, h, L, dk)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        out = self.wo.forward(out)

        self._cache = (q_in, k_in, v_in, Q, K, V, attn_softmax, attn, scores, mask, B, L, S)
        return out, attn

    def backward(self, dout):
        """dout: (B, L, d_model) - gradient from wo output."""
        q_in, k_in, v_in, Q, K, V, attn_softmax, attn, scores, mask, B, L, S = self._cache

        # Backward through wo
        dout_attn_out = self.wo.backward(dout)  # (B, L, d_model)

        # Un-merge heads
        dout_attn = dout_attn_out.reshape(B, L, self.h, self.dk).transpose(0, 2, 1, 3)

        # Backward through attention: out = attn @ V
        # d_attn = d_out @ V^T, d_V = attn^T @ d_out
        d_attn = dout_attn @ V.transpose(0, 1, 3, 2)  # (B, h, L, S) — grad w.r.t. post-dropout attn
        dV = attn.transpose(0, 1, 3, 2) @ dout_attn  # (B, h, S, dk)

        # Backward through attention dropout → grad w.r.t. pre-dropout attn (same shape)
        d_attn = self.attn_dropout.backward(d_attn)

        # Backward through softmax using PRE-dropout attention weights
        dscores = attn_softmax * (d_attn - (attn_softmax * d_attn).sum(axis=-1, keepdims=True))

        if mask is not None:
            dscores = np.where(mask, dscores, 0)

        dscores = dscores / math.sqrt(self.dk)

        # Backward through Q @ K^T: dQ = dscores @ K, dK = dscores^T @ Q
        dQ = dscores @ K  # (B, h, L, dk)
        dK = dscores.transpose(0, 1, 3, 2) @ Q  # (B, h, S, dk)

        # Merge heads back + backward through projections
        dQ = dQ.transpose(0, 2, 1, 3).reshape(B, L, self.d_model)
        dK = dK.transpose(0, 2, 1, 3).reshape(B, S, self.d_model)
        dV = dV.transpose(0, 2, 1, 3).reshape(B, S, self.d_model)

        dq_in = self.wq.backward(dQ)
        dk_in = self.wk.backward(dK)
        dv_in = self.wv.backward(dV)

        return dq_in, dk_in, dv_in


class PositionwiseFFN(Layer):
    """Section 3.3: FFN(x) = ReLU(xW1 + b1)W2 + b2"""
    def __init__(self, d_model=512, d_ff=2048):
        self.lin1 = Linear(d_model, d_ff)
        self.lin2 = Linear(d_ff, d_model)
        self._relu_out = None

    def forward(self, x):
        h = self.lin1.forward(x)
        self._relu_out = np.maximum(0, h)
        return self.lin2.forward(self._relu_out)

    def backward(self, dout):
        dh = self.lin2.backward(dout)
        dh[self._relu_out <= 0] = 0  # ReLU backward
        dx = self.lin1.backward(dh)
        return dx


# ═══════════════════════════════════════════════════════════════════════
#  TRANSFORMER BLOCKS
# ═══════════════════════════════════════════════════════════════════════

class EncoderLayer(Layer):
    def __init__(self, d_model=512, d_ff=2048, h=8, dropout=0.1):
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self._attn_store = None  # for capturing attention weights

    def forward(self, x, mask=None, training=True):
        attn_out, attn_w = self.self_attn.forward(x, x, x, mask, training=training)
        if self._attn_store is not None:
            self._attn_store.append(attn_w)  # (B, h, L, L)
        x = self.norm1.forward(x + self.dropout1.forward(attn_out, training=training))
        x = self.norm2.forward(x + self.dropout2.forward(self.ffn.forward(x), training=training))
        return x

    def backward(self, dout):
        # Backward norm2 + residual (FFN + dropout)
        dx2 = self.norm2.backward(dout)
        dffn = self.ffn.backward(self.dropout2.backward(dx2))
        dx2 += dffn  # residual

        # Backward norm1 + residual (attention + dropout)
        dx1 = self.norm1.backward(dx2)
        dattn = self.self_attn.backward(self.dropout1.backward(dx1))
        # residual: sum the three inputs to multi-head attention (q=k=v=x)
        dx = sum(dattn) if isinstance(dattn, tuple) else dattn
        dx1 += dx
        return dx1


class DecoderLayer(Layer):
    def __init__(self, d_model=512, d_ff=2048, h=8, dropout=0.1):
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
        self._d_enc_out = None  # accumulated encoder gradient
        self._attn_store = None  # for capturing attention weights

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None, training=True):
        self._enc_out = enc_out
        self._src_mask = src_mask
        self._d_enc_out = None

        attn_out, attn_w = self.self_attn.forward(x, x, x, tgt_mask, training=training)
        if self._attn_store is not None:
            self._attn_store.append(attn_w)  # decoder self-attention
        x = self.norm1.forward(x + self.dropout1.forward(attn_out, training=training))

        attn_out, attn_w = self.cross_attn.forward(x, enc_out, enc_out, src_mask, training=training)
        if self._attn_store is not None:
            self._attn_store.append(attn_w)  # decoder cross-attention
        x = self.norm2.forward(x + self.dropout2.forward(attn_out, training=training))

        x = self.norm3.forward(x + self.dropout3.forward(self.ffn.forward(x), training=training))
        return x

    def backward(self, dout):
        # norm3 + residual (FFN + dropout)
        dx3 = self.norm3.backward(dout)
        dffn = self.ffn.backward(self.dropout3.backward(dx3))
        dx3 += dffn

        # norm2 + residual (cross-attn + dropout)
        dx2 = self.norm2.backward(dx3)
        d_cross = self.cross_attn.backward(self.dropout2.backward(dx2))
        # d_cross = (dq, dk, dv). q comes from decoder, k,v from encoder.
        dx2 += d_cross[0]  # gradient w.r.t. q (decoder side)
        self._d_enc_out = d_cross[1] + d_cross[2]  # accumulate encoder gradient

        # norm1 + residual (self-attn + dropout)
        dx1 = self.norm1.backward(dx2)
        d_self = self.self_attn.backward(self.dropout1.backward(dx1))
        if isinstance(d_self, tuple):
            dx1 += d_self[0] + d_self[1] + d_self[2]
        else:
            dx1 += d_self
        return dx1


class Encoder(Layer):
    def __init__(self, vocab_size, d_model=512, N=6, d_ff=2048, h=8,
                 shared_weight=None, dropout=0.1):
        self.embed = Embedding(vocab_size, d_model, weight=shared_weight)
        self.embed_dropout = Dropout(dropout)
        self.layers = [EncoderLayer(d_model, d_ff, h, dropout) for _ in range(N)]

    def forward(self, x, mask=None, capture_attn=False, training=True):
        x = self.embed.forward(x) * math.sqrt(self.embed.w.shape[1])
        pe = self._make_pe(x.shape[1], x.shape[2])
        x = self.embed_dropout.forward(x + pe, training=training)
        if capture_attn:
            store = []
            for layer in self.layers:
                layer._attn_store = store
                x = layer.forward(x, mask, training=training)
                layer._attn_store = None
            return x, {'self_attn': store}
        for layer in self.layers:
            x = layer.forward(x, mask, training=training)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        # Backward through embedding dropout
        dout = self.embed_dropout.backward(dout)
        # Backward through embedding + PE (PE has no params)
        dx = self.embed.backward(dout)
        return dx

    def _make_pe(self, L, d_model):
        pe = np.zeros((L, d_model), dtype=np.float64)
        pos = np.arange(L, dtype=np.float64).reshape(-1, 1)
        div = np.exp(np.arange(0, d_model, 2, dtype=np.float64) *
                     (-math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        return pe[None, :, :]


class Decoder(Layer):
    def __init__(self, vocab_size, d_model=512, N=6, d_ff=2048, h=8,
                 shared_weight=None, dropout=0.1):
        self.embed = Embedding(vocab_size, d_model, weight=shared_weight)
        self.embed_dropout = Dropout(dropout)
        self.layers = [DecoderLayer(d_model, d_ff, h, dropout) for _ in range(N)]

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None,
                capture_attn=False, training=True):
        x = self.embed.forward(x) * math.sqrt(self.embed.w.shape[1])
        pe = self._make_pe(x.shape[1], x.shape[2])
        x = self.embed_dropout.forward(x + pe, training=training)
        if capture_attn:
            store = []
            for layer in self.layers:
                layer._attn_store = store
                x = layer.forward(x, enc_out, src_mask, tgt_mask, training=training)
                layer._attn_store = None
            # store: [self_attn_0, cross_attn_0, self_attn_1, cross_attn_1, ...]
            self_attn = store[0::2]   # even indices
            cross_attn = store[1::2]  # odd indices
            return x, {'self_attn': self_attn, 'cross_attn': cross_attn}
        for layer in self.layers:
            x = layer.forward(x, enc_out, src_mask, tgt_mask, training=training)
        return x

    def backward(self, dout):
        enc_grad = None
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
            if layer._d_enc_out is not None:
                if enc_grad is None:
                    enc_grad = layer._d_enc_out.copy()
                else:
                    enc_grad += layer._d_enc_out
        # Backward through embedding dropout
        dout = self.embed_dropout.backward(dout)
        dx = self.embed.backward(dout)
        return enc_grad  # return accumulated encoder gradient

    def _make_pe(self, L, d_model):
        pe = np.zeros((L, d_model), dtype=np.float64)
        pos = np.arange(L, dtype=np.float64).reshape(-1, 1)
        div = np.exp(np.arange(0, d_model, 2, dtype=np.float64) *
                     (-math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(pos * div)
        pe[:, 1::2] = np.cos(pos * div)
        return pe[None, :, :]


class Transformer(Layer):
    """Section 3: Full encoder-decoder with weight tying (Section 3.4)."""
    def __init__(self, src_vocab, tgt_vocab, d_model=512, N=6, d_ff=2048, h=8,
                 dropout=0.1):
        # Shared weight: one Param for encoder embed, decoder embed, and proj
        shared = Param(tgt_vocab, d_model)
        self.encoder = Encoder(src_vocab, d_model, N, d_ff, h,
                               shared_weight=shared, dropout=dropout)
        self.decoder = Decoder(tgt_vocab, d_model, N, d_ff, h,
                               shared_weight=shared, dropout=dropout)
        # proj uses shared weight transposed: y = x @ W^T + b
        self.proj = Linear(d_model, tgt_vocab, weight=shared, transpose_w=True)
        self._cache = None

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, training=True):
        enc_out = self.encoder.forward(src, src_mask, training=training)
        dec_out = self.decoder.forward(tgt, enc_out, src_mask, tgt_mask, training=training)
        logits = self.proj.forward(dec_out)
        self._cache = (src, tgt, src_mask, tgt_mask)
        return logits

    def forward_with_attention(self, src, tgt, src_mask=None, tgt_mask=None,
                               training=False):
        """Forward pass that also returns all attention weights.

        Returns:
            logits: (B, T, vocab) output logits
            attentions: dict with keys 'encoder' and 'decoder', each containing
                       lists of (B, h, seq, seq) attention weight arrays.
        """
        enc_out, enc_attn = self.encoder.forward(
            src, src_mask, capture_attn=True, training=training)
        dec_out, dec_attn = self.decoder.forward(
            tgt, enc_out, src_mask, tgt_mask, capture_attn=True, training=training)
        logits = self.proj.forward(dec_out)
        return logits, {'encoder': enc_attn, 'decoder': dec_attn}

    def backward(self, dlogits):
        ddec = self.proj.backward(dlogits)
        d_enc = self.decoder.backward(ddec)
        if d_enc is not None:
            self.encoder.backward(d_enc)

    def make_src_mask(self, src, pad_idx=0):
        return (src != pad_idx).astype(np.float64)[:, None, None, :]

    def make_tgt_mask(self, tgt):
        L = tgt.shape[1]
        return np.tril(np.ones((L, L), dtype=np.float64))[None, None, :, :]


# ═══════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════

class NoamLR:
    """Noam learning rate schedule (Section 5.3, Figure 2).

    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Linear warmup for first warmup_steps, then inverse sqrt decay.
    """
    def __init__(self, d_model=512, warmup_steps=4000, factor=1.0):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self._scale = d_model ** (-0.5)

    def __call__(self, step):
        arg1 = step ** (-0.5)
        arg2 = step * self.warmup_steps ** (-1.5)
        return self.factor * self._scale * min(arg1, arg2)


class Adam:
    """Adam optimizer (Section 5.3)."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), eps=1e-9):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g ** 2
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def clip_gradients(model, max_norm=5.0):
    """Clip global gradient norm to max_norm (Plan 05)."""
    total_norm = 0.0
    for p in model.params():
        total_norm += (p.grad ** 2).sum()
    total_norm = math.sqrt(total_norm)
    if total_norm > max_norm and total_norm > 0:
        scale = max_norm / total_norm
        for p in model.params():
            p.grad *= scale
    return total_norm


def save_checkpoint(model, optimizer, path, step=None, loss=None, extra=None):
    """Save model weights, optimizer state, and metadata (Plan 06)."""
    state = {'step': step, 'loss': loss}
    for i, p in enumerate(model.params()):
        state[f'param_{i}'] = p.data
        state[f'grad_{i}'] = p.grad
    if optimizer is not None:
        for i, (m, v) in enumerate(zip(optimizer.m, optimizer.v)):
            state[f'm_{i}'] = m
            state[f'v_{i}'] = v
        state['opt_t'] = optimizer.t
    if extra:
        state.update(extra)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savez_compressed(path, **state)


def load_checkpoint(model, optimizer, path):
    """Load model weights and optimizer state from a checkpoint."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = np.load(path, allow_pickle=True)
    for i, p in enumerate(model.params()):
        key = f'param_{i}'
        if key in state:
            p.data[:] = state[key]
            if f'grad_{i}' in state:
                p.grad[:] = state[f'grad_{i}']
    if optimizer is not None and 'opt_t' in state:
        for i, (m, v) in enumerate(zip(optimizer.m, optimizer.v)):
            key_m, key_v = f'm_{i}', f'v_{i}'
            if key_m in state:
                m[:] = state[key_m]
            if key_v in state:
                v[:] = state[key_v]
        optimizer.t = int(state['opt_t'])
    return state


def softmax(x, axis=-1):
    x_max = x.max(axis=axis, keepdims=True)
    exp = np.exp(x - x_max)
    return exp / exp.sum(axis=axis, keepdims=True)


def cross_entropy_loss(logits, targets, smoothing=0.1, ignore_index=0):
    """Cross-entropy with label smoothing (Section 5.4)."""
    N, C = logits.shape
    probs = softmax(logits, axis=-1)
    log_probs = np.log(probs + 1e-9)

    mask = targets != ignore_index
    valid = np.where(mask)[0]

    if len(valid) == 0:
        return 0.0, np.zeros_like(logits)

    # Label smoothing
    smoothed = np.full((N, C), smoothing / C)
    smoothed[np.arange(N), targets] += 1.0 - smoothing

    loss = -(smoothed * log_probs).sum(axis=1)[mask].mean()

    # Gradient: dL/dlogits = probs - smoothed_targets (for valid positions)
    dlogits = (probs - smoothed) / max(len(valid), 1)
    dlogits[~mask] = 0

    return loss, dlogits


def make_copy_batch(vocab_size, seq_len, batch_size, bos_idx=1, eos_idx=2):
    data = np.random.randint(3, vocab_size, (batch_size, seq_len)).astype(np.int64)
    src = data.copy()
    tgt_in = np.concatenate([np.full((batch_size, 1), bos_idx), data], axis=1)
    tgt_out = np.concatenate([data, np.full((batch_size, 1), eos_idx)], axis=1)
    return src, tgt_in, tgt_out


def train_copy(model, steps=500, batch_size=16, seq_len=5, vocab_size=20,
               lr=1e-3, log_every=100, use_noam=False, clip_norm=0.0,
               d_model=32, warmup_steps=100, checkpoint_dir=None,
               save_every=9999):
    """
    Train on copy task with optional NoamLR, gradient clipping, checkpointing.

    Args:
        model: Transformer instance
        use_noam: if True, use NoamLR schedule instead of fixed LR
        clip_norm: max gradient norm (0 = no clipping)
        checkpoint_dir: if set, save checkpoints here
        save_every: save checkpoint every N steps
    """
    opt = Adam(model.params(), lr=lr)
    lr_schedule = NoamLR(d_model=d_model, warmup_steps=warmup_steps) if use_noam else None
    start_step = 0

    for step in range(start_step + 1, steps + 1):
        if use_noam:
            opt.lr = lr_schedule(step)

        src_np, tgt_in_np, tgt_out_np = make_copy_batch(vocab_size, seq_len, batch_size)

        src_mask = model.make_src_mask(src_np)
        tgt_mask = model.make_tgt_mask(tgt_in_np)

        logits = model.forward(src_np, tgt_in_np, src_mask, tgt_mask, training=True)
        B, L, C = logits.shape

        loss, dlogits = cross_entropy_loss(
            logits.reshape(-1, C), tgt_out_np.reshape(-1),
            smoothing=0.1)

        model.zero_grad()
        model.backward(dlogits.reshape(B, L, C))

        grad_norm = clip_gradients(model, max_norm=clip_norm) if clip_norm > 0 else 0.0
        opt.step()

        if step % log_every == 0 or step == 1:
            preds = logits.argmax(axis=-1)
            correct = (preds == tgt_out_np).sum()
            total = tgt_out_np.size
            acc = correct / total * 100
            gn = f"  grad_norm {grad_norm:.2f}" if clip_norm > 0 else ""
            print(f"step {step:4d}  loss {loss:.4f}  acc {acc:.1f}%  "
                  f"lr {opt.lr:.2e}{gn}")

        if checkpoint_dir and step % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step:06d}.npz')
            save_checkpoint(model, opt, ckpt_path, step=step, loss=loss)
            # Also overwrite latest
            latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.npz')
            save_checkpoint(model, opt, latest_path, step=step, loss=loss)

    return model


def train_wmt(model, dataset, steps=10000, batch_size=32, d_model=512,
               warmup_steps=4000, log_every=100, eval_every=500,
               tokenizer=None, dev_dataset=None, clip_norm=0.0,
               checkpoint_dir=None, save_every=1000):
    """
    Training loop for parallel text with NoamLR, clipping, and checkpointing.

    Args:
        clip_norm: max gradient norm (0 = no clipping)
        checkpoint_dir: if set, save checkpoints here
        save_every: save checkpoint every N steps
    """
    opt = Adam(model.params(), lr=1.0)  # lr overridden by NoamLR
    lr_schedule = NoamLR(d_model=d_model, warmup_steps=warmup_steps)

    # Resume from checkpoint if available
    start_step = 0
    if checkpoint_dir:
        latest = os.path.join(checkpoint_dir, 'checkpoint_latest.npz')
        if os.path.exists(latest):
            print(f"  Resuming from checkpoint: {latest}")
            state = load_checkpoint(model, opt, latest)
            start_step = int(state.get('step', 0))
            print(f"  Resumed at step {start_step}")

    step = start_step
    while step < steps:
        for src, tgt_in, tgt_out in dataset.batches(batch_size=batch_size):
            step += 1
            opt.lr = lr_schedule(step)

            src_mask = model.make_src_mask(src)
            tgt_mask = model.make_tgt_mask(tgt_in)

            logits = model.forward(src, tgt_in, src_mask, tgt_mask, training=True)
            B, L, C = logits.shape

            loss, dlogits = cross_entropy_loss(
                logits.reshape(-1, C), tgt_out.reshape(-1),
                smoothing=0.1)

            model.zero_grad()
            model.backward(dlogits.reshape(B, L, C))

            grad_norm = clip_gradients(model, max_norm=clip_norm) if clip_norm > 0 else 0.0
            opt.step()

            if step % log_every == 0 or step == 1:
                preds = logits.argmax(axis=-1)
                mask = tgt_out != 0
                correct = ((preds == tgt_out) & mask).sum()
                total = mask.sum()
                acc = correct / max(total, 1) * 100
                gn = f"  grad_norm {grad_norm:.2f}" if clip_norm > 0 else ""
                print(f"step {step:6d}  loss {loss:.4f}  acc {acc:.1f}%  "
                      f"lr {opt.lr:.2e}{gn}")

            if step % eval_every == 0 and tokenizer is not None and dev_dataset is not None:
                _eval_bleu(model, dev_dataset, tokenizer, step)

            if checkpoint_dir and step % save_every == 0:
                ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_step_{step:06d}.npz')
                save_checkpoint(model, opt, ckpt_path, step=step, loss=loss)
                latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.npz')
                save_checkpoint(model, opt, latest_path, step=step, loss=loss)

            if step >= steps:
                break

    # Save final checkpoint
    if checkpoint_dir:
        final_path = os.path.join(checkpoint_dir, 'checkpoint_final.npz')
        save_checkpoint(model, opt, final_path, step=step, loss=loss)
        print(f"  Saved final checkpoint: {final_path}")

    return model


def _eval_bleu(model, dataset, tokenizer, step, max_samples=50):
    """Translate dev set and compute BLEU."""
    from data import corpus_bleu

    hypotheses = []
    references = []
    count = 0

    for src_batch, tgt_in_batch, tgt_out_batch in dataset.batches(batch_size=8, shuffle=False):
        for i in range(src_batch.shape[0]):
            if count >= max_samples:
                break
            src_tokens = src_batch[i]
            # Strip padding
            src_tokens = src_tokens[src_tokens != 0]
            # Use greedy decoding (inference mode = no dropout)
            out_tokens = translate(model, src_tokens, bos_idx=2, eos_idx=3)

            hyp_text = tokenizer.decode(out_tokens)
            ref_tokens = tgt_out_batch[i]
            ref_tokens = ref_tokens[ref_tokens != 0]
            ref_text = tokenizer.decode(ref_tokens)

            hypotheses.append(hyp_text)
            references.append(ref_text)
            count += 1
        if count >= max_samples:
            break

    if hypotheses:
        result = corpus_bleu(hypotheses, references)
        print(f"  >>> step {step:6d}  dev BLEU {result['score']:.2f}  "
              f"({len(hypotheses)} sentences)")


def translate(model, src_np, max_len=30, bos_idx=1, eos_idx=2):
    """Greedy autoregressive decoding (inference mode, no dropout)."""
    src_mask = model.make_src_mask(src_np[None, :])
    enc_out = model.encoder.forward(src_np[None, :], src_mask, training=False)

    tgt = np.array([[bos_idx]], dtype=np.int64)
    for _ in range(max_len):
        tgt_mask = model.make_tgt_mask(tgt)
        dec_out = model.decoder.forward(tgt, enc_out, src_mask, tgt_mask, training=False)
        logits = model.proj.forward(dec_out)
        next_token = logits[0, -1].argmax()
        tgt = np.concatenate([tgt, [[next_token]]], axis=1)
        if next_token == eos_idx:
            break
    return tgt[0]


# ═══════════════════════════════════════════════════════════════════════
#  BEAM SEARCH (Section 5.5 / Section 6.1)
# ═══════════════════════════════════════════════════════════════════════

def translate_beam(model, src_np, beam_size=4, max_len=30, alpha=0.6,
                   bos_idx=1, eos_idx=2):
    """
    Beam search decoding (Section 5.5). Inference mode = no dropout.

    Maintains `beam_size` candidate hypotheses. At each step, expands every
    hypothesis by the full vocabulary, then prunes to the top `beam_size`
    by length-normalized score (Section 6.1):  score = log_prob / (len ** alpha)

    Encoder runs once; decoder runs on the full beam in parallel.
    """
    src_mask = model.make_src_mask(src_np[None, :])
    enc_out = model.encoder.forward(src_np[None, :], src_mask, training=False)

    # Each beam: (tokens list, score)
    beams = [([bos_idx], 0.0)]

    for _ in range(max_len):
        # Separate finished from active beams
        finished = [b for b in beams if b[0][-1] == eos_idx]
        active = [b for b in beams if b[0][-1] != eos_idx]
        if not active:
            break

        B = len(active)
        max_b_len = max(len(b[0]) for b in active)

        # Build padded batch: (B, max_b_len)
        tgt_batch = np.zeros((B, max_b_len), dtype=np.int64)
        for i, (toks, _) in enumerate(active):
            tgt_batch[i, :len(toks)] = toks

        # Batched decoder forward (inference mode)
        tgt_mask = model.make_tgt_mask(tgt_batch)
        enc_batch = np.repeat(enc_out, B, axis=0)
        logits = model.proj.forward(
            model.decoder.forward(tgt_batch, enc_batch, src_mask, tgt_mask,
                                  training=False))
        last_lp = np.log(softmax(logits[:, -1, :], axis=-1) + 1e-9)

        # Expand: each active beam -> vocab_size candidates, keep top beam_size
        candidates = []
        for i, (toks, score) in enumerate(active):
            for v in range(last_lp.shape[1]):
                new_score = score + last_lp[i, v]
                candidates.append((toks + [v], new_score))

        # Sort by length-normalized score
        candidates.sort(key=lambda c: c[1] / ((len(c[0])) ** alpha), reverse=True)

        # Keep top beam_size, merge with finished
        beams = finished + candidates[:beam_size - len(finished)]
        beams = beams[:beam_size]

    # Return highest-scoring (by length-normalized score)
    beams.sort(key=lambda b: b[1] / (len(b[0]) ** alpha), reverse=True)
    return np.array(beams[0][0], dtype=np.int64)


def translate_with_beam(model, src_np, beam_size=4, max_len=30, alpha=0.6,
                        bos_idx=1, eos_idx=2):
    """
    Convenience wrapper: runs greedy if beam_size=1, beam search otherwise.
    Returns output tokens and the score.
    """
    if beam_size <= 1:
        tokens = translate(model, src_np, max_len, bos_idx, eos_idx)
        return tokens, 0.0
    tokens = translate_beam(model, src_np, beam_size, max_len, alpha, bos_idx, eos_idx)
    return tokens, 0.0


# ═══════════════════════════════════════════════════════════════════════
#  DEMO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    np.random.seed(42)

    print("=" * 50)
    print("Transformer — Pure NumPy with Manual Backprop")
    print("=" * 50)

    # Paper base model size
    base = Transformer(37000, 37000, d_model=512, N=6, d_ff=2048, h=8)
    print(f"\nPaper base model: {base.param_count():,} params  (paper: ~65M)")

    # Small model for training
    VOCAB = 20
    SEQ_LEN = 4

    model = Transformer(VOCAB, VOCAB, d_model=32, N=2, d_ff=64, h=4, dropout=0)
    print(f"Train model:     {model.param_count():,} params\n")

    # Before training
    print("Before training:")
    for i in range(2):
        src = np.random.randint(3, VOCAB, (SEQ_LEN,))
        out = translate(model, src)
        correct = src.tolist() == out[1:1+SEQ_LEN].tolist()
        print(f"  src: {src.tolist()}  out: {out.tolist()}  {'OK' if correct else 'WRONG'}")

    t0 = time.time()
    model = train_copy(model, steps=300, batch_size=16, seq_len=SEQ_LEN,
                       vocab_size=VOCAB, log_every=75)
    print(f"\nTraining time: {time.time() - t0:.1f}s")

    # After training — compare greedy vs beam search
    print("\nAfter training:")
    greedy_ok = 0
    beam_ok = 0
    for i in range(10):
        src = np.random.randint(3, VOCAB, (SEQ_LEN,))

        out_greedy = translate(model, src)
        ok_g = src.tolist() == out_greedy[1:1+SEQ_LEN].tolist()
        greedy_ok += ok_g

        out_beam = translate_beam(model, src, beam_size=4)
        ok_b = src.tolist() == out_beam[1:1+SEQ_LEN].tolist()
        beam_ok += ok_b

        if ok_g != ok_b:
            print(f"  DIFF  src={src.tolist()}")
            print(f"         greedy: {out_greedy.tolist()}  {'OK' if ok_g else 'WRONG'}")
            print(f"         beam4:  {out_beam.tolist()}   {'OK' if ok_b else 'WRONG'}")
        else:
            status = "OK" if ok_g else "WRONG"
            print(f"  src={src.tolist()}  greedy={out_greedy.tolist()}  beam4={out_beam.tolist()}  {status}")
    print(f"\nGreedy accuracy: {greedy_ok}/10")
    print(f"Beam4  accuracy: {beam_ok}/10")
