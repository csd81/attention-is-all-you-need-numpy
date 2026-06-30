# WMT Data Training

**Goal:** Train on real WMT English↔German sentence pairs and measure BLEU score.

## Why

The copy task proves the model *works*, but it's not real translation. Training on WMT 2014 EN-DE (4.5M sentence pairs) with BPE tokenization is the path to:
- Real BLEU scores
- Comparison with published results
- Actual usability

## Prerequisites

### Data: WMT 2014 EN-DE

Available via `torchtext` or direct download from:
```
https://data.statmt.org/wmt17/translation-task/
```

Files needed:
- `train.tok.clean.bpe.32000.en`
- `train.tok.clean.bpe.32000.de`
- `newstest2013.tok.bpe.32000.en` (dev)
- `newstest2014.tok.bpe.32000.en` (test)
- Corresponding `.de` files

### Tokenization: BPE (Section 5.1)

The paper uses:
- BPE encoding with ~37,000 shared source/target tokens (EN-DE)
- Word-piece with 32,000 tokens (EN-FR)
- We use `sentencepiece` or `subword-nmt` library

### Data pipeline

```python
class WMTDataset:
    def __init__(self, src_path, tgt_path, batch_size):
        # Load sentences
        # BPE tokenize
        # Numericalize
        # Bucket by length for efficient batching

    def batch(self, batch_size):
        # Group similar-length sequences
        # Pad to max length in batch
        # Return src, tgt_in, tgt_out tensors
```

## Training Config (Paper Section 5.2)

### Base model (Table 3)
| Hyperparameter | Value |
|---|---|
| d_model | 512 |
| d_ff | 2048 |
| h | 8 |
| N | 6 |
| Dropout | 0.1 |
| Label smoothing | 0.1 |
| warmup_steps | 4000 |
| Batch (tokens) | ~25,000 src + 25,000 tgt |
| Steps | 100,000 |
| Hardware | 8 × P100 GPUs |
| Time | ~12 hours |

Our config (CPU-friendly):
```python
d_model=128, N=3, h=4, d_ff=512, batch_size=32
```
Smaller model, more steps.

### Optimizer (Section 5.3)

```
lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
```
- Linear warmup for first 4K steps
- Then inverse sqrt decay

### BLEU scoring

Use `sacrebleu` library:
```python
import sacrebleu
bleu = sacrebleu.corpus_bleu(hypotheses, [references])
```

Paper results to beat:
- Base model: 27.3 BLEU (EN-DE)
- Big model: 28.4 BLEU (EN-DE)

## Implementation steps

1. Download and prepare WMT 2014 data → ~100MB download
2. Train BPE tokenizer with 37K merges
3. Build batched dataset with bucketed padding
4. Update training loop for variable-length sequences
5. Implement BLEU evaluation callback
6. Train for 100K steps (or smaller proxy)
7. Translate newstest2014 and report BLEU

## Expected timeline

| Component | Time |
|---|---|
| Data download + preprocessing | ~15 min (download) |
| BPE training | ~5 min |
| Training (small model, CPU) | ~hours |
| BLEU evaluation | <1 min |
| Full paper-spec training | needs GPU |

## Tests

- `test_dataset_length` — WMT dataset returns expected number of sentence pairs
- `test_bpe_tokenization` — BPE encodes and decodes without OOV tokens
- `test_batch_shapes` — bucketed batches have correct (B, L) shapes
- `test_batch_no_padding_collapse` — not every batch is all padding
- `test_bleu_perfect_match` — BLEU=100 when hypothesis == reference
- `test_bleu_no_match` — BLEU=0 when hypothesis shares no n-grams with reference
- `test_train_step` — single training step completes without error
