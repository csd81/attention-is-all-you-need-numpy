# Noam Learning Rate Schedule + Gradient Clipping

**Goal:** Implement the Transformer-specific LR schedule from Section 5.3 and add gradient clipping.

## Why

The Noam schedule is critical for Transformer training — it's not optional. The paper states:

```
lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```

- **Linear warmup** (first 4K steps): Prevents the model from making huge updates when parameters are randomly initialized and gradients are noisy
- **Inverse sqrt decay** (after warmup): Gradually reduces the LR for fine-grained convergence

Without this schedule, the Transformer is noticeably harder to train. Our current fixed LR works on the toy copy task but would struggle on real data.

Gradient clipping (`max_norm=1.0` or `5.0`) is complementary — prevents any single batch from destabilizing training.

## How

### NoamLR class

```python
class NoamLR:
    """Section 5.3: lrate = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)"""
    def __init__(self, optimizer, d_model=512, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5))
        for p in self.optimizer.params:
            # store current lr for logging
            pass
        self.optimizer.lr = lr
        self.optimizer.step()
```

### Gradient clipping

```python
def clip_gradients(model, max_norm=1.0):
    total_norm = 0.0
    for p in model.params():
        total_norm += (p.grad ** 2).sum()
    total_norm = math.sqrt(total_norm)
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for p in model.params():
            p.grad *= scale
    return total_norm
```

### Changes to training loop

```python
scheduler = NoamLR(optimizer, d_model=model.encoder.embed.w.shape[1])

for step in range(steps):
    # forward + backward
    grad_norm = clip_gradients(model, max_norm=5.0)
    scheduler.step()
```

### Diagnostic: log the LR

```
step   500  loss 4.12  lr 1.23e-04  grad_norm 2.45
step  1000  loss 3.01  lr 8.67e-05  grad_norm 1.89
```

This lets you verify the schedule is working and detect training issues early.

## Changes needed

- Add `NoamLR` class (~15 lines)
- Add `clip_gradients` function (~10 lines)
- Modify `train_copy` to use both (~5 lines)
- Log LR and grad_norm in training output (~3 lines)

Nothing in the model architecture changes — purely a training loop improvement.

## Tests

- `test_noam_lr_warmup` — LR increases linearly for the first `warmup_steps`
- `test_noam_lr_decay` — LR decreases after `warmup_steps`
- `test_noam_lr_d_model_scaling` — larger d_model gives lower peak LR
- `test_gradient_clip_norm` — after clipping, global gradient norm <= max_norm
- `test_gradient_clip_direction` — clipped gradients point in the same direction as unclipped
- `test_training_converges` — copy task still reaches 100% accuracy with new schedule
