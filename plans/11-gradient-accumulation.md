# Gradient Accumulation

## Context

The paper uses large batch sizes but on a laptop, memory limits how many
sentences fit in a single batch. Gradient accumulation simulates a larger
batch by running multiple forward/backward passes (summing gradients) before
each optimizer step.

## Change

**File:** `transformer.py`

Added `accum_steps=1` parameter to both `train_copy` and `train_wmt`.

When `accum_steps > 1`:
1. Inner loop runs `accum_steps` micro-batches per optimizer step
2. Gradients accumulate naturally (no `zero_grad()` between micro-batches)
3. After accumulation, gradients are divided by `accum_steps` (to get the mean)
4. One `clip_gradients()` + `opt.step()` follows

The log reports averaged loss/accuracy across micro-batches.

For `train_wmt`, micro-batches are consumed from the same dataset iterator
(via `next()` inside the inner loop), so bucketing is preserved.

## Verification

- `test_grad_accum_training`: trains with `accum_steps=2`, verifies output
- `test_grad_accum_matches`: `accum_steps=1` is deterministic with same seed
- 41/41 tests pass
