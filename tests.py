"""
Unit tests for the Transformer implementation.
"""
import sys
import math
import numpy as np

sys.path.insert(0, '.')
from transformer import (
    Transformer, translate, translate_beam, train_copy, train_wmt,
    make_copy_batch, Param, softmax, NoamLR, Adam,
)
from data import BPETokenizer, ParallelDataset, corpus_bleu, sentence_bleu, BOS_ID, EOS_ID, PAD_ID, UNK_ID


# ═══════════════════════════════════════════════════════════════════════
#  WEIGHT TYING (plan 02)
# ═══════════════════════════════════════════════════════════════════════

def test_param_count_drops():
    """Weight tying: param count matches shared-weight expectation."""
    m = Transformer(37000, 37000, d_model=512, N=6, d_ff=2048, h=8)
    assert m.param_count() < 70_000_000, f"Too many params: {m.param_count():,}"


def test_weight_identity():
    """Weight tying: all three layers share the same Param object."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4)
    assert m.encoder.embed.w is m.decoder.embed.w, "encoder/decode embed not shared"
    assert m.encoder.embed.w is m.proj.w, "embed/proj not shared"


def test_gradient_accumulation():
    """Weight tying: gradients accumulate into the same array."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4)
    src = np.random.randint(3, 20, (2, 4)).astype(np.int64)
    tgt_in = np.concatenate([np.full((2, 1), 1), src], axis=1)
    tgt_out = np.concatenate([src, np.full((2, 1), 2)], axis=1)

    src_mask = m.make_src_mask(src)
    tgt_mask = m.make_tgt_mask(tgt_in)
    logits = m.forward(src, tgt_in, src_mask, tgt_mask)

    from transformer import cross_entropy_loss
    loss, dlogits = cross_entropy_loss(logits.reshape(-1, 20), tgt_out.reshape(-1))
    m.zero_grad()
    m.backward(dlogits.reshape(2, 5, 20))

    # Verify same gradient memory
    assert m.encoder.embed.w.grad is m.decoder.embed.w.grad
    assert m.encoder.embed.w.grad is m.proj.w.grad
    assert not np.all(m.encoder.embed.w.grad == 0), "no gradient accumulated"


def test_training_still_works():
    """Weight tying: copy task still reaches high accuracy."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4)
    np.random.seed(42)
    m = train_copy(m, steps=300, batch_size=16, seq_len=4, vocab_size=20, log_every=999)
    out = translate(m, np.array([5, 8, 3, 12]))
    assert out[1:5].tolist() == [5, 8, 3, 12], f"copy failed: {out.tolist()}"


# ═══════════════════════════════════════════════════════════════════════
#  BEAM SEARCH (plan 01)
# ═══════════════════════════════════════════════════════════════════════

def test_beam_size_1_equals_greedy():
    """Beam search: beam=1 produces same output as greedy."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4)
    np.random.seed(42)
    m = train_copy(m, steps=200, batch_size=16, seq_len=4, vocab_size=20, log_every=999)
    src = np.array([5, 8, 3, 12])
    g = translate(m, src)
    b = translate_beam(m, src, beam_size=1)
    assert g.tolist() == b.tolist(), f"greedy={g.tolist()} != beam1={b.tolist()}"


def test_beam_terminates():
    """Beam search: always returns within max_len tokens."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4)
    for _ in range(5):
        src = np.random.randint(3, 20, (4,))
        out = translate_beam(m, src, max_len=30)
        assert len(out) <= 31, f"too many tokens: {len(out)}"  # BOS + max_len
        assert len(out) >= 1, "empty output"


def test_beam_reproduces_input():
    """Beam search: trained copy model copies correctly."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4)
    np.random.seed(42)
    m = train_copy(m, steps=200, batch_size=16, seq_len=4, vocab_size=20, log_every=999)
    out = translate_beam(m, np.array([5, 8, 3, 12]), beam_size=4)
    assert out[1:5].tolist() == [5, 8, 3, 12], f"beam copy failed: {out.tolist()}"


# ═══════════════════════════════════════════════════════════════════════
#  ATTENTION VISUALIZATION (plan 03)
# ═══════════════════════════════════════════════════════════════════════

def test_attention_shapes():
    """Attention: all 3 types return correct shapes per layer/head."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4)
    src = np.random.randint(3, 20, (1, 4)).astype(np.int64)
    tgt_in = np.array([[1, 5, 8, 3, 12]], dtype=np.int64)
    src_mask = m.make_src_mask(src)
    tgt_mask = m.make_tgt_mask(tgt_in)
    logits, attn = m.forward_with_attention(src, tgt_in, src_mask, tgt_mask)

    enc_self = attn['encoder']['self_attn']
    dec_self = attn['decoder']['self_attn']
    dec_cross = attn['decoder']['cross_attn']

    assert len(enc_self) == 2, f"expected 2 layers, got {len(enc_self)}"
    assert enc_self[0].shape == (1, 4, 4, 4), f"enc self shape: {enc_self[0].shape}"
    assert dec_self[0].shape[1] == 4, f"dec self heads: {dec_self[0].shape}"
    assert dec_cross[0].shape[-1] == 4, f"cross attn src dim: {dec_cross[0].shape}"


def test_attention_weights_sum_to_one():
    """Attention: each row of the attention matrix sums to 1.0."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4)
    src = np.random.randint(3, 20, (1, 4)).astype(np.int64)
    tgt_in = np.array([[1, 5, 8, 3, 12]], dtype=np.int64)
    src_mask = m.make_src_mask(src)
    tgt_mask = m.make_tgt_mask(tgt_in)
    logits, attn = m.forward_with_attention(src, tgt_in, src_mask, tgt_mask)

    for layer_key in ['encoder', 'decoder']:
        for attn_key in ['self_attn', 'cross_attn']:
            for w in attn[layer_key].get(attn_key, []):
                row_sums = w[0, 0].sum(axis=-1)  # first batch, first head
                assert np.allclose(row_sums, 1.0, atol=1e-5), \
                    f"row sums: {row_sums}"


def test_causal_mask():
    """Attention: decoder self-attention has zeros in the upper triangle."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4)
    src = np.random.randint(3, 20, (1, 4)).astype(np.int64)
    tgt_in = np.array([[1, 5, 8, 3, 12]], dtype=np.int64)
    src_mask = m.make_src_mask(src)
    tgt_mask = m.make_tgt_mask(tgt_in)
    logits, attn = m.forward_with_attention(src, tgt_in, src_mask, tgt_mask)

    w = attn['decoder']['self_attn'][0][0, 0]  # layer 0, head 0, batch 0
    L = w.shape[0]
    for i in range(L):
        for j in range(i + 1, L):
            assert w[i, j] == 0.0, f"pos {i} attends to future pos {j}: {w[i,j]}"


def test_no_attention_to_padding():
    """Attention: masked positions have exactly 0 attention weight."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4)
    # Source with trailing PAD (index 0)
    src = np.array([[5, 8, 0, 0]], dtype=np.int64)
    tgt_in = np.array([[1, 5, 8, 2]], dtype=np.int64)
    src_mask = m.make_src_mask(src)
    tgt_mask = m.make_tgt_mask(tgt_in)
    logits, attn = m.forward_with_attention(src, tgt_in, src_mask, tgt_mask)

    # Encoder self-attention should not attend to PAD positions
    for w in attn['encoder']['self_attn']:
        attn_to_pad = w[0, :, :, 2:].sum()  # positions 2,3 are PAD
        assert attn_to_pad < 1e-6, f"attention to pad: {attn_to_pad}"


def test_visualize_no_crash():
    """Visualization: runs without error on a small model."""
    from visualize import visualize_all, _tokens_to_str, plot_attention_grid
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4)
    np.random.seed(42)
    m = train_copy(m, steps=100, batch_size=8, seq_len=4, vocab_size=20, log_every=999)
    src = np.array([[5, 8, 3, 12]], dtype=np.int64)
    tgt_in = np.array([[1, 5, 8, 3, 12]], dtype=np.int64)
    src_tok = _tokens_to_str(src[0].tolist())
    tgt_tok = _tokens_to_str(tgt_in[0].tolist())
    figs, logits, _ = visualize_all(m, src, tgt_in, src_tok, tgt_tok)
    assert len(figs) == 3, f"expected 3 figures, got {len(figs)}"
    import matplotlib.pyplot as plt
    plt.close('all')


# ═══════════════════════════════════════════════════════════════════════
#  WMT DATA PIPELINE (plan 04)
# ═══════════════════════════════════════════════════════════════════════

def _make_test_tokenizer():
    """Train a tiny BPE tokenizer on the test data."""
    tok = BPETokenizer()
    tok.train('test_data/train.en', 'test_data/train.de',
              model_prefix='test_data/bpe_test', vocab_size=200)
    return tok


def test_bpe_tokenization():
    """BPE: encodes and decodes without OOV tokens."""
    tok = _make_test_tokenizer()
    text = "the cat sat on the mat"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    # Should decode back to similar text (whitespace may differ)
    assert len(ids) > 0, "empty encoding"
    assert UNK_ID not in ids, f"OOV token in {ids.tolist()}"
    assert len(decoded) > 0, "empty decode"


def test_bpe_roundtrip():
    """BPE: encode_sentence includes BOS/EOS tokens."""
    tok = _make_test_tokenizer()
    ids = tok.encode_sentence("hello", bos=True, eos=True)
    assert ids[0] == BOS_ID, f"expected BOS=2 at start, got {ids[0]}"
    assert ids[-1] == EOS_ID, f"expected EOS=3 at end, got {ids[-1]}"
    assert len(ids) >= 3, f"too short: {ids.tolist()}"


def test_dataset_length():
    """Dataset: returns expected number of sentence pairs."""
    tok = _make_test_tokenizer()
    ds = ParallelDataset('test_data/train.en', 'test_data/train.de',
                         tok, max_len=50)
    assert ds.n_pairs == 10, f"expected 10 pairs, got {ds.n_pairs}"


def test_batch_shapes():
    """Dataset: bucketed batches have correct (B, L) shapes."""
    tok = _make_test_tokenizer()
    ds = ParallelDataset('test_data/train.en', 'test_data/train.de',
                         tok, max_len=50)
    for i, (src, tgt_in, tgt_out) in enumerate(ds.batches(batch_size=4, shuffle=False)):
        B = src.shape[0]
        assert B <= 4, f"batch too large: {B}"
        assert src.ndim == 2, f"src shape: {src.shape}"
        assert tgt_in.ndim == 2, f"tgt_in shape: {tgt_in.shape}"
        assert tgt_out.ndim == 2, f"tgt_out shape: {tgt_out.shape}"
        assert tgt_in.shape == tgt_out.shape, \
            f"tgt_in {tgt_in.shape} != tgt_out {tgt_out.shape}"
        # Verify padding at position 0 is PAD_ID
        if i == 0:
            break  # one batch is enough


def test_batch_no_all_padding():
    """Dataset: not every batch is all padding."""
    tok = _make_test_tokenizer()
    ds = ParallelDataset('test_data/train.en', 'test_data/train.de',
                         tok, max_len=50)
    all_pad = True
    for src, tgt_in, tgt_out in ds.batches(batch_size=4, shuffle=False):
        if (src != 0).sum() > 0:
            all_pad = False
            break
    assert not all_pad, "all batches are all-padding"


def test_bleu_perfect_match():
    """BLEU: score 100 when hypothesis == reference."""
    result = corpus_bleu(
        ["the cat sat on the mat"],
        ["the cat sat on the mat"])
    assert result['score'] > 99, f"perfect BLEU too low: {result['score']}"


def test_bleu_no_match():
    """BLEU: score 0 when hypothesis shares no 4-grams."""
    result = corpus_bleu(
        ["goodbye world"],
        ["the cat sat on the mat"])
    assert result['score'] < 1, f"no-match BLEU too high: {result['score']}"


def test_train_step():
    """Training: single WMT training step completes without error."""
    tok = _make_test_tokenizer()
    ds = ParallelDataset('test_data/train.en', 'test_data/train.de',
                         tok, max_len=50)
    model = Transformer(tok.vocab_size, tok.vocab_size,
                        d_model=32, N=2, d_ff=64, h=4)
    np.random.seed(42)
    model = train_wmt(model, ds, steps=3, batch_size=4, d_model=32,
                       warmup_steps=2, log_every=999, eval_every=999)
    assert model is not None, "train_wmt returned None"


def test_noam_lr_schedule():
    """NoamLR: produces expected learning rate values."""
    schedule = NoamLR(d_model=512, warmup_steps=4000, factor=1.0)
    lr1 = schedule(1)
    lr2 = schedule(4000)
    lr3 = schedule(8000)
    assert lr1 > 0, f"LR at step 1 is {lr1}"
    assert lr2 > lr1, f"LR should increase during warmup: {lr1} -> {lr2}"
    assert lr3 < lr2, f"LR should decay after warmup: {lr2} -> {lr3}"
    # At step=warmup_steps, both formulas give the same value
    # lr = d_model^(-0.5) * warmup_steps^(-0.5)
    expected_at_warmup = 512 ** (-0.5) * 4000 ** (-0.5)
    assert abs(lr2 - expected_at_warmup) < 1e-10, \
        f"LR at warmup: {lr2} != {expected_at_warmup}"


# ═══════════════════════════════════════════════════════════════════════
#  RUN ALL
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    np.random.seed(42)
    tests = [
        ('test_param_count_drops', test_param_count_drops),
        ('test_weight_identity', test_weight_identity),
        ('test_gradient_accumulation', test_gradient_accumulation),
        ('test_training_still_works', test_training_still_works),
        ('test_beam_size_1_equals_greedy', test_beam_size_1_equals_greedy),
        ('test_beam_terminates', test_beam_terminates),
        ('test_beam_reproduces_input', test_beam_reproduces_input),
        ('test_attention_shapes', test_attention_shapes),
        ('test_attention_weights_sum_to_one', test_attention_weights_sum_to_one),
        ('test_causal_mask', test_causal_mask),
        ('test_no_attention_to_padding', test_no_attention_to_padding),
        ('test_visualize_no_crash', test_visualize_no_crash),
        # WMT data pipeline (plan 04)
        ('test_bpe_tokenization', test_bpe_tokenization),
        ('test_bpe_roundtrip', test_bpe_roundtrip),
        ('test_dataset_length', test_dataset_length),
        ('test_batch_shapes', test_batch_shapes),
        ('test_batch_no_all_padding', test_batch_no_all_padding),
        ('test_bleu_perfect_match', test_bleu_perfect_match),
        ('test_bleu_no_match', test_bleu_no_match),
        ('test_train_step', test_train_step),
        ('test_noam_lr_schedule', test_noam_lr_schedule),
    ]

    passed = 0
    failed = []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f'  PASS  {name}')
        except Exception as e:
            failed.append(name)
            print(f'  FAIL  {name}: {e}')

    total = len(tests)
    print(f'\n{passed}/{total} passed')
    if failed:
        print(f'FAILED: {failed}')
        sys.exit(1)
    else:
        print('All tests passed!')
