# Checkpointing (Save/Load Model)

**Goal:** Save and restore model parameters and optimizer state so training can be interrupted and resumed.

## Why

- Training a real Transformer takes hours or days — you need to checkpoint or risk losing all progress
- Lets you run multiple training experiments and compare results
- Enables model distribution: train once, deploy elsewhere
- Essential prerequisite for the WMT training plan

## How

### Save format: NumPy `.npz`

Each `Param` is saved as a key-value pair. Simple, portable, no dependencies:

```python
def save_checkpoint(model, optimizer, path, step=None, loss=None, extra=None):
    state = {'step': step, 'loss': loss}
    for i, p in enumerate(model.params()):
        state[f'param_{i}'] = p.data
        state[f'grad_{i}'] = p.grad
    if optimizer:
        for i, (m, v) in enumerate(zip(optimizer.m, optimizer.v)):
            state[f'm_{i}'] = m
            state[f'v_{i}'] = v
        state['opt_t'] = optimizer.t
    if extra:
        state.update(extra)
    np.savez_compressed(path, **state)
```

### Load

```python
def load_checkpoint(model, optimizer, path):
    state = np.load(path)
    for i, p in enumerate(model.params()):
        p.data[:] = state[f'param_{i}']
        p.grad[:] = state[f'grad_{i}']
    if optimizer and 'opt_t' in state:
        for i, (m, v) in enumerate(zip(optimizer.m, optimizer.v)):
            m[:] = state[f'm_{i}']
            v[:] = state[f'v_{i}']
        optimizer.t = int(state['opt_t'])
    return state
```

### Training loop integration

```python
start_step = 0
if resume_path and os.path.exists(resume_path):
    state = load_checkpoint(model, opt, resume_path)
    start_step = int(state['step'])

for step in range(start_step + 1, total_steps + 1):
    ...
    if step % save_every == 0:
        save_checkpoint(model, opt, f'checkpoint_step_{step}.npz',
                        step=step, loss=loss)
```

### Checkpoint naming strategy

| File | Contents |
|---|---|
| `checkpoint_latest.npz` | Overwritten each save — always recoverable |
| `checkpoint_step_05000.npz` | Periodic snapshots for rollback |
| `checkpoint_best.npz` | Saved when validation loss is lowest |

### File size estimate

Paper base model (65M params, float64): ~500 MB per checkpoint.
Our small demo model (45K params): ~350 KB.

Use `np.float32` to halve storage: change `Param` init to `np.float32`, or convert on save.

## Changes needed

- Add `save_checkpoint()` and `load_checkpoint()` functions
- Modify training loop to save every N steps and on completion
- Add `--resume` CLI argument to training script
- ~40 lines of new code

## Tests

- `test_save_and_load_params` — loaded model produces identical forward pass as saved model
- `test_save_and_load_optimizer` — optimizer state (m, v, t) is identical after save/load cycle
- `test_resume_training` — training for 100 steps, saving, resuming, and training 100 more steps produces same loss as training 200 steps straight
- `test_checkpoint_file_exists` — checkpoint file is created on disk after save
- `test_load_nonexistent_raises` — loading a missing file raises an appropriate error
- `test_checkpoint_roundtrip_dtype` — float64 params remain float64 after save/load
