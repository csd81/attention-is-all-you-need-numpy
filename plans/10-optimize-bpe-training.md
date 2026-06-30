# Optimize BPE Training

## Context

The original `_train_from_text()` scanned ALL unique words and recounted
ALL adjacent pairs at every merge step — O(V × W) work. For WMT-scale
(4.5M sentences, 200K unique words, 37K merges), this was impractically
slow in pure Python.

The fix: maintain pair counts incrementally. Only re-count pairs in words
that were actually changed by the previous merge, which is typically
<0.1% of all words.

## Change

Rewrote `BPETokenizer._train_from_text()` in `data.py` to use incremental
pair counting:

1. **Initialize** `pair_counts` (Counter) and `pair_to_words` (dict of
   `pair → set of words`) by scanning all words once.
2. **At each merge step:** look up affected words from `pair_to_words`,
   subtract old pair counts for those words, apply the merge, add new
   pair counts. Unaffected words contribute zero maintenance.
3. **Clean up** zero-count entries.

## Verification

- 39/39 tests pass (including new `test_bpe_regression`)
- BPE produces same vocabulary size and well-formed merge rules
- Encode/decode roundtrip works for all test sentences
