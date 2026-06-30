# Weight Tying

**Goal:** Share weights between encoder embedding, decoder embedding, and pre-softmax projection (Section 3.4).

## Why

The paper states: "In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]."

This:
- Reduces parameter count by ~40% (65M → reported 65M — currently we have 101M because we don't share)
- Acts as a regularizer
- The embedding matrices are huge (vocab × d_model), so sharing them is a massive memory saving
- Reference: Press & Wolf (2016) "Using the Output Embedding to Improve Language Models"

## How

Currently we have three separate weight matrices:
- `encoder.embed.w` — Param(vocab, d_model)
- `decoder.embed.w` — Param(vocab, d_model)
- `proj.w` — Param(d_model, vocab) [transposed]

### Option A: Hard tying (recommended)

Create a single shared Param:
```python
self.shared_weight = Param(vocab, d_model)
```

Then in forward passes:
- `encoder.embed.forward(x)` = `self.shared_weight[x]`
- `decoder.embed.forward(x)` = `self.shared_weight[x]`
- `proj.forward(x)` = `x @ self.shared_weight.T + self.b`

This ensures gradients are accumulated into the same parameter during backward.

### Option B: Soft tying with L2 penalty

Keep separate weights but add `||W_enc - W_dec||^2 + ||W_dec - W_proj.T||^2` to the loss.
Inferior in practice — use Option A.

### Changes to Transformer

In `__init__`:
- Create shared Param
- Pass it to encoder, decoder, and proj (they accept it via constructor)

In `forward`:
- All three layers reference the same Tensor

In `params()`:
- Only list `shared_weight` once (don't double-count)

## Impact

Paper base model:
- Before: 101M params
- After: ~63M params (matches paper's 65M closely)
- Quality: small improvement on perplexity/BLEU

## Tests

- `test_param_count_drops` — param count matches shared-weight expectation
- `test_forward_pass` — forward pass produces same-shaped output as before
- `test_gradients_accumulate` — encoder embed, decoder embed, and proj all accumulate gradients into the same shared weight
- `test_weight_identity` — `encoder.embed.w.data` is the same object as `decoder.embed.w.data` (not a copy)
- `test_training_still_works` — copy task still reaches 100% accuracy
