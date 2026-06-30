# Beam Search

**Goal:** Replace greedy decoding with beam search (beam=4, length penalty α=0.6 per Section 6.1).

## Why

Greedy decoding picks the single most probable token at each step, which can lead to locally optimal but globally poor sequences. Beam search maintains K candidate hypotheses, dramatically improving BLEU. The paper used beam search for all reported results.

## How

### BeamState
- Maintain `beam_size` hypotheses, each as a list of token IDs
- Each hypothesis has a cumulative log-probability score
- Normalize by sequence length with length penalty

### Search loop
```
1. Encode source → enc_out (single forward, cached)
2. Initialize beam with [BOS], score=0
3. At each step, run decoder on all beam hypotheses in parallel
4. Get top-K logits from the last token of each hypothesis
5. Select top beam_size candidates across ALL hypotheses
6. Prune: keep only the best scoring partial sequences
7. Stop condition:
   - All beams hit [EOS]
   - OR max length reached
```

### Parallel decode
- Pad all beam hypotheses to the same length
- Run a single batched decoder forward instead of looping over beams
- Mask future positions correctly for each hypothesis's length

### Length penalty (Section 6.1)
```
score = log_prob / (len ** alpha)    # alpha = 0.6
```
Shorter sequences get a boost, preventing the model from preferring overly short outputs.

### Return
- Hypothesis with the highest normalized score
- Optionally return the full beam (for reranking or logging)

## Changes needed

- New class `BeamSearch` in the inference section
- Modify `translate()` to have `beam_size` parameter
- ~50 lines of code

## Tests

- `test_beam_size_1_equals_greedy` — beam=1 produces same output as greedy decode
- `test_beam_small_vocab` — beam search on a tiny vocab (size 3) returns valid tokens
- `test_beam_reproduces_input` — on a trained copy model, beam=4 copies correctly
- `test_beam_terminates` — always returns within max_len tokens (no infinite loop)
- `test_beam_length_penalty_effect` — alpha=0 and alpha=1 produce different rankings
- `test_beam_single_step` — max_len=1 returns [BOS, token]
