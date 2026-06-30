# Four Features: Norm Logging, Eval Mode, Benchmark

## Changes

### 1. Per-Layer Gradient Norm Logging (`transformer.py`)

Added `log_norms` parameter to `train_copy` and `train_wmt`. When `True`,
logs gradient L2 norms for every major component:

```
grad  dec_0_self_attn 0.0000  dec_embed 1.1198  enc_0_norm1 0.0176  ...
```

Helpers: `_compute_grad_norms(model)`, `_euclidean_norm(params)`,
`_layer_params(layer)`, `_format_norms(norms)`.

### 2. Per-Layer Parameter Norm Tracking (`transformer.py`)

Same structure as gradient norms but for `p.data`. `_compute_param_norms(model)`
uses `_euclidean_norm(params, attr='data')`. Both logged in the same line:

```
grad  ...  param  dec_0_norm1 5.6579  ...
```

### 3. Inference Mode (`transformer.py`)

Added `train(mode=True)` and `eval()` to `Transformer`. `forward()` now
defaults `training=None` and falls back to `self._training`.

Sub-layers keep their `training` parameter (no cascade change).

### 4. Training Speed Benchmark (`benchmark.py`)

New standalone script. Results:

```
     model        params     ms/step
------------------------------------
      tiny        43,412       5.430
     small     1,019,592      40.857
      base    63,119,496     981.092
```

## Files Modified

- `transformer.py`: norm logging, train/eval mode, helpers
- `tests.py`: 6 new tests (train/eval, norm logging, benchmark)
- `benchmark.py`: new file

## Verification

- 47/47 tests pass
- Demo converges to 9/10 accuracy
- Benchmark produces timing table
- Norm logging shows per-layer grad + param norms in training output
