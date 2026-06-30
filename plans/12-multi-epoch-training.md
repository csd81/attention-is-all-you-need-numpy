# Multi-Epoch WMT Training

## Context

`train_wmt` already re-creates the dataset iterator when exhausted via a
`while step < steps` outer loop, giving implicit multi-epoch behavior.
But there was no visibility into which epoch the training was in.

## Change

**File:** `transformer.py` — `train_wmt` function

Added `epoch` counter and included it in the log output:

```
epoch 1  step      1  loss 5.3081  ppl 201.97  acc 0.0%  lr 6.25e-02
```

## Verification

- 41/41 tests pass
- `test_train_step` output shows `epoch 1` prefix
