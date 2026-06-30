# Perplexity Reporting

## Context

All Transformer paper results report perplexity as a key metric alongside
BLEU. Perplexity = exp(loss) and measures how "perplexed" the model is —
lower is better, 1.0 is perfect.

The training logs only showed `loss` and `acc`, making it hard to compare
against paper tables.

## Change

**File:** `transformer.py`

Added `ppl = math.exp(loss)` and included `ppl` in both `train_copy` and
`train_wmt` log lines. The log format went from:

    step    1  loss 3.0048  acc 2.5%  lr 1.00e-03

to:

    step    1  loss 3.0048  ppl 20.18  acc 2.5%  lr 1.00e-03

## Verification

1. `python tests.py` — all 38 tests pass
2. `python transformer.py` — copy task shows ppl dropping from ~20 to ~2
