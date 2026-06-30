# Vectorize Embedding Backward

## Context

The `Embedding.backward()` method at `transformer.py:180` uses nested Python for-loops to
accumulate gradients. For real WMT training with B=32, L=100, vocab=37000, this loop
executes 3200 Python iterations per step, each doing an indexed add into a large gradient
matrix. This becomes a major bottleneck.

## Change

Replace nested for-loops with `np.add.at` — a single C-level call that handles duplicate
indices correctly.

**Before:**
```python
def backward(self, dout):
    for i in range(self._indices.shape[0]):
        for j in range(self._indices.shape[1]):
            idx = self._indices[i, j]
            self.w.grad[idx] += dout[i, j]
    return None
```

**After:**
```python
def backward(self, dout):
    np.add.at(self.w.grad, self._indices, dout)
    return None
```

## Verification

1. `python tests.py` — all 36 tests pass
2. `python transformer.py` — copy task trains to ~9/10 accuracy
