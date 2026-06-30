# Attention Visualization

**Goal:** Extract and plot attention weights from all heads to see what the model learns (Section 4).

## Why

The paper notes: "Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences."

Visualizing attention heads is the best way to build intuition for *why* the Transformer works:
- Which words attend to which other words
- How different heads specialize (e.g., positional, syntactic relations)
- How attention patterns evolve through layers

## How

### Step 1: Extract attention weights

Modify `ScaledDotProductAttention` (or the forward passes) to return attention matrices:

For each of the 3 attention types:
- **Encoder self-attention**: tokens → tokens (6 layers × 8 heads = 48 maps)
- **Decoder self-attention**: tokens → previous tokens (masked, 48 maps)
- **Decoder cross-attention**: tokens → encoder tokens (48 maps)

### Step 2: Plot with matplotlib

Create a `visualize.py` module that generates:

**Heatmap grid** (most useful):
```python
fig, axes = plt.subplots(6, 8, figsize=(16, 12))
for layer in range(6):
    for head in range(8):
        axes[layer, head].imshow(attn_map[layer, head], cmap='viridis')
        axes[layer, head].axis('off')
```

- X-axis: positions attended to
- Y-axis: positions attending
- Color intensity = attention weight

**Token-labeled heatmaps**:
For single sentences, label axes with actual tokens:
```python
axes.imshow(attn_map)
axes.set_xticks(range(len(tokens)))
axes.set_xticklabels(tokens, rotation=90)
axes.set_yticks(range(len(tokens)))
axes.set_yticklabels(tokens)
```

### Step 3: Patterns to look for

| Pattern | Meaning |
|---|---|
| Strong diagonal | Each token attends mostly to itself |
| Off-diagonal bands | Syntactic relations (verb→subject, etc.) |
| Uniform columns | [CLS]-like token attending everywhere |
| Block patterns | Phrase-level grouping |

### Step 4: Multi-head comparison

- Create a figure where each head is one subplot
- Title each subplot with head index
- Check for specialization across heads

### Step 5: Cross-attention patterns

Cross-attention (decoder→encoder) is particularly interesting:
- Which encoder tokens does each decoder position focus on?
- Does the model align source/target words correctly?

## Changes needed

- Modify forward passes to store attention weights
- New file: `visualize.py` (depends on matplotlib)
- `translate_with_attention()` function that returns both outputs and attention maps
- ~80 lines of new code

## Tests

- `test_attention_shapes` — all 3 attention types return correct shapes per layer/head
- `test_attention_weights_sum_to_one` — each row of the attention matrix sums to 1.0
- `test_causal_mask_visible` — decoder self-attention has zeros in the upper triangle
- `test_no_attention_to_padding` — masked positions have exactly 0 attention weight
- `test_visualize_no_crash` — `visualize()` runs without error on a small model
