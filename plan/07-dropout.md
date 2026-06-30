# Dropout (Section 5.4)

**Goal:** Add dropout in the correct positions to match the paper's regularization setup.

## Why

The paper uses dropout in three places (Section 5.4):

1. **Output of each sub-layer, before residual add**: `LayerNorm(x + Dropout(Sublayer(x)))`
2. **Embedding + positional encoding sums**: `Dropout(Embedding(x) * sqrt(d_model) + PE)`
3. **Attention weights**: `Dropout(softmax(...))` before `@ V`

Our current implementation has **zero dropout** — the `dropout=0.1` parameter exists in the constructor signatures but is never used. On the copy task this doesn't matter, but on real data the model will overfit badly without it.

The paper found dropout to be "very helpful in avoiding over-fitting" (Table 3, row D: removing dropout drops BLEU from 25.8 → 24.6).

## How

### Dropout layer

```python
class Dropout(Layer):
    def __init__(self, rate=0.1):
        self.rate = rate
        self.mask = None

    def forward(self, x, training=True):
        if not training or self.rate == 0:
            return x
        self.mask = np.random.binomial(1, 1 - self.rate, x.shape).astype(np.float64)
        return x * self.mask / (1 - self.rate)

    def backward(self, dout):
        if self.mask is None:
            return dout
        return dout * self.mask / (1 - self.rate)
```

### Placement in the architecture

**Encoder/Decoder layers** (Section 5.4 Residual Dropout):
```
x = norm1(x + dropout1(self_attn(x, ...)))
x = norm2(x + dropout2(ffn(x)))
```

**Embedding** (Section 5.4):
```
x = dropout(embed(x) * sqrt(d_model) + pe)
```

**Attention** (Section 5.4):
```
attn = dropout(softmax(scores))
```

### Changes needed

- Add `Dropout` class (~15 lines)
- Add dropout fields in `Encoder`, `Decoder`, `EncoderLayer`, `DecoderLayer`, `MultiHeadAttention`
- Pass `training=True/False` flag through forward passes (model.train() / model.eval())
- Toggle dropout off during inference (greedy/beam decoding)
- ~50 lines of new code across 7 files/classes

The `training` flag propagation is the most invasive change — every `forward()` call chain needs to accept and forward it, or the model class sets a global mode variable.

## Tests

- `test_dropout_training_vs_eval` — forward pass gives different outputs in train vs eval mode
- `test_dropout_eval_deterministic` — same input in eval mode always gives same output
- `test_dropout_rate_zero` — rate=0 behaves identically to no dropout
- `test_dropout_expected_mask_rate` — with rate=0.1, ~10% of values are zeroed (average over many passes)
- `test_dropout_backward` — backward pass completes without error in training mode
- `test_dropout_no_leak_eval` — gradients are correct when dropout is disabled during eval/test
