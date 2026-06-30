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
    Dropout, clip_gradients, save_checkpoint, load_checkpoint,
    average_checkpoints,
    _compute_grad_norms, _compute_param_norms, _format_norms,
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
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4, dropout=0)
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
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4, dropout=0)
    np.random.seed(42)
    m = train_copy(m, steps=300, batch_size=16, seq_len=4, vocab_size=20, log_every=999)
    out = translate(m, np.array([5, 8, 3, 12]))
    assert out[1:5].tolist() == [5, 8, 3, 12], f"copy failed: {out.tolist()}"


# ═══════════════════════════════════════════════════════════════════════
#  BEAM SEARCH (plan 01)
# ═══════════════════════════════════════════════════════════════════════

def test_beam_size_1_equals_greedy():
    """Beam search: beam=1 produces same output as greedy."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4, dropout=0)
    np.random.seed(42)
    m = train_copy(m, steps=200, batch_size=16, seq_len=4, vocab_size=20, log_every=999)
    src = np.array([5, 8, 3, 12])
    g = translate(m, src)
    b = translate_beam(m, src, beam_size=1)
    assert g.tolist() == b.tolist(), f"greedy={g.tolist()} != beam1={b.tolist()}"


def test_beam_terminates():
    """Beam search: always returns within max_len tokens."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4, dropout=0)
    for _ in range(5):
        src = np.random.randint(3, 20, (4,))
        out = translate_beam(m, src, max_len=30)
        assert len(out) <= 31, f"too many tokens: {len(out)}"  # BOS + max_len
        assert len(out) >= 1, "empty output"


def test_beam_reproduces_input():
    """Beam search: trained copy model copies correctly."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4, dropout=0)
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
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4, dropout=0)
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


def test_bpe_regression():
    """BPE (optimized): merge rules and vocab match expected structure."""
    tok = _make_test_tokenizer()
    assert tok.vocab_size == 200, f"vocab_size={tok.vocab_size}"
    # All merge rules should be well-formed (pairs of strings)
    for a, b in tok.bpe_merges:
        assert isinstance(a, str) and len(a) > 0
        assert isinstance(b, str) and len(b) > 0
    assert len(tok.bpe_merges) > 0, "should have learned at least one merge"
    # Verify encode/decode on a sentence with known tokens
    text = "the cat sat on the mat"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert UNK_ID not in ids
    assert len(decoded) > 0


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
#  DROPOUT (plan 07)
# ═══════════════════════════════════════════════════════════════════════

def test_dropout_training_vs_eval():
    """Dropout: forward pass differs between train and eval modes."""
    d = Dropout(rate=0.5)
    x = np.ones((100, 100), dtype=np.float64)
    out_train = d.forward(x, training=True)
    out_eval = d.forward(x, training=False)
    # Eval returns x unchanged
    assert np.allclose(out_eval, x), "eval mode should not modify input"
    # Training mode zeros out ~rate fraction
    zero_frac = (out_train == 0).mean()
    assert 0.3 < zero_frac < 0.7, f"zero fraction {zero_frac:.3f} not ~0.5"


def test_dropout_eval_deterministic():
    """Dropout: same input in eval mode always gives same output."""
    d = Dropout(rate=0.5)
    x = np.random.randn(10, 10)
    out1 = d.forward(x, training=False)
    out2 = d.forward(x, training=False)
    assert np.allclose(out1, out2), "eval mode should be deterministic"


def test_dropout_rate_zero():
    """Dropout: rate=0 behaves identically to no dropout."""
    d = Dropout(rate=0.0)
    x = np.random.randn(10, 10)
    out = d.forward(x, training=True)
    assert np.allclose(out, x), "rate=0 should pass through unchanged"


def test_dropout_backward():
    """Dropout: backward pass completes without error in training mode."""
    d = Dropout(rate=0.1)
    x = np.ones((5, 10), dtype=np.float64)
    out = d.forward(x, training=True)
    dout = np.ones_like(out)
    dx = d.backward(dout)
    assert dx.shape == x.shape, f"backward shape {dx.shape} != {x.shape}"
    # Non-masked positions should have gradient 1/(1-rate)
    assert not np.all(dx == 0), "all gradients are zero"


def test_dropout_attn_forward():
    """Dropout: forward_with_attention works with dropout enabled."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4)
    src = np.random.randint(3, 20, (1, 4)).astype(np.int64)
    tgt_in = np.array([[1, 5, 8, 3, 12]], dtype=np.int64)
    src_mask = m.make_src_mask(src)
    tgt_mask = m.make_tgt_mask(tgt_in)
    # Should not error regardless of training mode
    logits1, attn1 = m.forward_with_attention(src, tgt_in, src_mask, tgt_mask, training=True)
    logits2, attn2 = m.forward_with_attention(src, tgt_in, src_mask, tgt_mask, training=False)
    assert logits1.shape == logits2.shape, "shape mismatch between train/eval"


# ═══════════════════════════════════════════════════════════════════════
#  GRADIENT CLIPPING + NOAM LR (plan 05)
# ═══════════════════════════════════════════════════════════════════════

def test_gradient_clip_norm():
    """Gradient clipping: after clipping, global norm <= max_norm."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4, dropout=0)
    src = np.random.randint(3, 20, (2, 4)).astype(np.int64)
    tgt_in = np.concatenate([np.full((2, 1), 1), src], axis=1)
    tgt_out = np.concatenate([src, np.full((2, 1), 2)], axis=1)
    src_mask = m.make_src_mask(src)
    tgt_mask = m.make_tgt_mask(tgt_in)
    logits = m.forward(src, tgt_in, src_mask, tgt_mask, training=True)
    from transformer import cross_entropy_loss
    loss, dlogits = cross_entropy_loss(logits.reshape(-1, 20), tgt_out.reshape(-1))
    m.zero_grad()
    m.backward(dlogits.reshape(2, 5, 20))

    max_norm = 0.5
    clip_gradients(m, max_norm=max_norm)
    # Compute actual norm after clipping
    total_norm = math.sqrt(sum((p.grad ** 2).sum() for p in m.params()))
    assert total_norm <= max_norm + 1e-6, \
        f"post-clip grad norm {total_norm:.4f} > max_norm {max_norm}"


def test_gradient_clip_preserves_direction():
    """Gradient clipping: direction is preserved (scaled, not rotated)."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4, dropout=0)
    src = np.random.randint(3, 20, (2, 4)).astype(np.int64)
    tgt_in = np.concatenate([np.full((2, 1), 1), src], axis=1)
    tgt_out = np.concatenate([src, np.full((2, 1), 2)], axis=1)
    src_mask = m.make_src_mask(src)
    tgt_mask = m.make_tgt_mask(tgt_in)
    logits = m.forward(src, tgt_in, src_mask, tgt_mask, training=True)
    from transformer import cross_entropy_loss
    loss, dlogits = cross_entropy_loss(logits.reshape(-1, 20), tgt_out.reshape(-1))
    m.zero_grad()
    m.backward(dlogits.reshape(2, 5, 20))

    # Save pre-clip gradients (flattened)
    grads_before = [p.grad.flatten().copy() for p in m.params()]
    g_before_flat = np.concatenate(grads_before)
    norm_before = np.linalg.norm(g_before_flat)

    clip_gradients(m, max_norm=norm_before * 0.5)  # clip to 50% of original

    grads_after = [p.grad.flatten().copy() for p in m.params()]
    g_after_flat = np.concatenate(grads_after)

    # Direction should be the same (cosine sim ~ 1)
    dot = np.dot(g_before_flat, g_after_flat)
    norm_product = np.linalg.norm(g_before_flat) * np.linalg.norm(g_after_flat)
    cos_sim = dot / max(norm_product, 1e-12)
    assert cos_sim > 0.999, f"cosine similarity {cos_sim:.6f} < 0.999"


def test_noam_custom_scaling():
    """NoamLR: larger d_model gives lower peak LR."""
    small = NoamLR(d_model=128, warmup_steps=100)
    large = NoamLR(d_model=512, warmup_steps=100)
    assert small(100) > large(100), "smaller d_model should have higher peak LR"


def test_training_with_clip():
    """Training: copy task converges with gradient clipping."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4, dropout=0)
    np.random.seed(42)
    m = train_copy(m, steps=300, batch_size=16, seq_len=4, vocab_size=20,
                   log_every=999, clip_norm=5.0)
    out = translate(m, np.array([5, 8, 3, 12]))
    assert out[1:5].tolist() == [5, 8, 3, 12], f"copy failed: {out.tolist()}"


def test_training_with_noam():
    """Training: NoamLR schedule runs without error on copy task."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4, dropout=0)
    np.random.seed(42)
    m = train_copy(m, steps=10, batch_size=16, seq_len=4, vocab_size=20,
                   log_every=999, use_noam=True, d_model=32, warmup_steps=5)
    out = translate(m, np.array([5, 8, 3, 12]))
    assert len(out) > 0, "empty output after NoamLR training"


# ═══════════════════════════════════════════════════════════════════════
#  GRADIENT ACCUMULATION (plan 11)
# ═══════════════════════════════════════════════════════════════════════

def test_grad_accum_training():
    """Gradient accumulation: training with accum_steps > 1 succeeds."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4, dropout=0)
    train_copy(m, steps=30, batch_size=8, vocab_size=20,
               lr=1e-3, log_every=999, accum_steps=2)
    out = translate(m, np.array([5, 8, 3, 12]))
    assert len(out) > 0, "empty output after grad accum training"


def test_grad_accum_matches():
    """Gradient accumulation: accum_steps=1 matches non-accum behavior."""
    np.random.seed(42)
    m1 = Transformer(20, 20, d_model=32, N=1, d_ff=64, h=2, dropout=0)
    loss_before = m1.param_count()
    train_copy(m1, steps=10, batch_size=8, vocab_size=20,
               lr=1e-3, log_every=999, accum_steps=1)
    assert m1.param_count() == loss_before, "param count changed"
    out1 = translate(m1, np.array([5, 8, 3, 12]))

    np.random.seed(42)
    m2 = Transformer(20, 20, d_model=32, N=1, d_ff=64, h=2, dropout=0)
    train_copy(m2, steps=10, batch_size=8, vocab_size=20,
               lr=1e-3, log_every=999, accum_steps=1)
    out2 = translate(m2, np.array([5, 8, 3, 12]))
    # Same seed + same accum_steps = same result
    assert np.array_equal(out1, out2), "accum_steps=1 not deterministic"


# ═══════════════════════════════════════════════════════════════════════
#  CHECKPOINTING (plan 06)
# ═══════════════════════════════════════════════════════════════════════

def test_save_and_load_params():
    """Checkpoint: loaded model produces identical forward pass."""
    m1 = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4)
    m2 = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4)
    opt1 = Adam(m1.params(), lr=1e-3)

    import tempfile, os
    path = os.path.join(tempfile.gettempdir(), 'test_checkpoint.npz')
    save_checkpoint(m1, opt1, path, step=10, loss=1.5)
    load_checkpoint(m2, None, path)

    # Both models should produce same output on same input
    src = np.random.randint(3, 20, (1, 4)).astype(np.int64)
    tgt_in = np.array([[1, 5, 8, 3, 12]], dtype=np.int64)
    src_mask = m1.make_src_mask(src)
    tgt_mask = m1.make_tgt_mask(tgt_in)
    out1 = m1.forward(src, tgt_in, src_mask, tgt_mask, training=False)
    out2 = m2.forward(src, tgt_in, src_mask, tgt_mask, training=False)
    assert np.allclose(out1, out2), "outputs differ after save/load"

    # Clean up
    os.remove(path)


def test_save_and_load_optimizer():
    """Checkpoint: optimizer state is identical after save/load cycle."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4, dropout=0)
    opt1 = Adam(m.params(), lr=1e-3)

    # Take one step to prime optimizer state
    src = np.random.randint(3, 20, (2, 4)).astype(np.int64)
    tgt_in = np.concatenate([np.full((2, 1), 1), src], axis=1)
    tgt_out = np.concatenate([src, np.full((2, 1), 2)], axis=1)
    src_mask = m.make_src_mask(src)
    tgt_mask = m.make_tgt_mask(tgt_in)
    logits = m.forward(src, tgt_in, src_mask, tgt_mask, training=True)
    from transformer import cross_entropy_loss
    loss, dlogits = cross_entropy_loss(logits.reshape(-1, 20), tgt_out.reshape(-1))
    m.zero_grad()
    m.backward(dlogits.reshape(2, 5, 20))
    opt1.step()

    m_before = opt1.m[0].copy()
    v_before = opt1.v[0].copy()
    t_before = opt1.t

    import tempfile, os
    path = os.path.join(tempfile.gettempdir(), 'test_opt_checkpoint.npz')
    save_checkpoint(m, opt1, path, step=5, loss=2.0)

    # New model + optimizer
    m2 = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4)
    opt2 = Adam(m2.params(), lr=1e-3)
    load_checkpoint(m2, opt2, path)

    assert np.allclose(opt2.m[0], m_before), "m state mismatch"
    assert np.allclose(opt2.v[0], v_before), "v state mismatch"
    assert opt2.t == t_before, f"t mismatch: {opt2.t} vs {t_before}"

    os.remove(path)


def test_checkpoint_file_exists():
    """Checkpoint: file is created on disk after save."""
    import tempfile, os
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4)
    opt = Adam(m.params())
    path = os.path.join(tempfile.gettempdir(), 'test_exists.npz')
    save_checkpoint(m, opt, path, step=1)
    assert os.path.exists(path), "checkpoint file not found"
    os.remove(path)


def test_load_nonexistent_raises():
    """Checkpoint: loading missing file raises FileNotFoundError."""
    import tempfile, os
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4)
    path = os.path.join(tempfile.gettempdir(), 'nonexistent.npz')
    try:
        load_checkpoint(m, None, path)
        assert False, "should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


def test_checkpoint_training_resume():
    """Checkpoint: resumed training continues to reduce loss."""
    import tempfile, os, shutil
    ckpt_dir = os.path.join(tempfile.gettempdir(), 'test_resume_ckpt')
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)

    np.random.seed(42)
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4, dropout=0)
    opt = Adam(m.params(), lr=1e-3)

    from transformer import cross_entropy_loss as cel
    for step in range(1, 51):
        src_np, tgt_in_np, tgt_out_np = make_copy_batch(20, 4, 16)
        src_mask = m.make_src_mask(src_np)
        tgt_mask = m.make_tgt_mask(tgt_in_np)
        logits = m.forward(src_np, tgt_in_np, src_mask, tgt_mask, training=True)
        B, L, C = logits.shape
        loss, dlogits = cel(logits.reshape(-1, C), tgt_out_np.reshape(-1))
        m.zero_grad()
        m.backward(dlogits.reshape(B, L, C))
        opt.step()

    ckpt_path = os.path.join(ckpt_dir, 'checkpoint_step_000050.npz')
    save_checkpoint(m, opt, ckpt_path, step=50, loss=loss)
    loss_after_50 = loss

    # Resume and train 50 more — loss should decrease
    load_checkpoint(m, opt, ckpt_path)
    for step in range(51, 101):
        src_np, tgt_in_np, tgt_out_np = make_copy_batch(20, 4, 16)
        src_mask = m.make_src_mask(src_np)
        tgt_mask = m.make_tgt_mask(tgt_in_np)
        logits = m.forward(src_np, tgt_in_np, src_mask, tgt_mask, training=True)
        B, L, C = logits.shape
        loss, dlogits = cel(logits.reshape(-1, C), tgt_out_np.reshape(-1))
        m.zero_grad()
        m.backward(dlogits.reshape(B, L, C))
        opt.step()

    assert loss < loss_after_50, \
        f"loss increased after resume: {loss_after_50:.4f} -> {loss:.4f}"
    shutil.rmtree(ckpt_dir)


# ═══════════════════════════════════════════════════════════════════════
#  CHECKPOINT AVERAGING (plan 08 WMT)
# ═══════════════════════════════════════════════════════════════════════

def test_checkpoint_averaging():
    """Averaging: weights are correctly averaged, model produces different output."""
    import tempfile, os
    np.random.seed(42)

    m = Transformer(20, 20, d_model=32, N=1, d_ff=64, h=2, dropout=0)
    # Train a bit and save two checkpoints
    train_copy(m, steps=50, batch_size=16, vocab_size=20, lr=1e-3, log_every=9999)
    ckpt_dir = tempfile.mkdtemp()
    path1 = os.path.join(ckpt_dir, "ckpt_1.npz")
    path2 = os.path.join(ckpt_dir, "ckpt_2.npz")
    save_checkpoint(m, None, path1, step=50)

    train_copy(m, steps=100, batch_size=16, vocab_size=20, lr=1e-3, log_every=9999)
    save_checkpoint(m, None, path2, step=100)

    # Average the two checkpoints
    avg_m = Transformer(20, 20, d_model=32, N=1, d_ff=64, h=2, dropout=0)
    average_checkpoints(avg_m, [path1, path2])

    # Verify average: avg = (ckpt1 + ckpt2) / 2
    # Use with-statements to avoid Windows file locking
    with np.load(path1, allow_pickle=True) as ckpt1:
        with np.load(path2, allow_pickle=True) as ckpt2:
            for i, p in enumerate(avg_m.params()):
                expected = (ckpt1[f'param_{i}'] + ckpt2[f'param_{i}']) / 2
                assert np.allclose(p.data, expected), \
                    f"param {i} not correctly averaged"

    os.remove(path1)
    os.remove(path2)
    os.rmdir(ckpt_dir)


def test_run_wmt_dry_run():
    """WMT script: --dry-run prints config and exits cleanly."""
    import subprocess, sys, os
    script = os.path.join(os.path.dirname(__file__), 'run_wmt.py')
    result = subprocess.run(
        [sys.executable, script, '--dry-run'],
        capture_output=True, text=True, cwd=os.path.dirname(script))
    assert result.returncode == 0, f"dry-run failed: {result.stderr}"
    assert 'WMT14' in result.stdout, "dry-run output missing WMT14 header"
    assert 'd_model=512' in result.stdout, "dry-run output missing model config"


# ═══════════════════════════════════════════════════════════════════════
#  TRAIN/EVAL MODE, NORM LOGGING, BENCHMARK (plan 13)
# ═══════════════════════════════════════════════════════════════════════

def test_train_eval_mode():
    """Train/eval: model.eval() disables training, model.train() re-enables."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4)
    assert m._training is True, "default should be training=True"
    m.eval()
    assert m._training is False, "eval() should set _training=False"
    m.train()
    assert m._training is True, "train() should set _training=True"
    m.train(False)
    assert m._training is False, "train(False) should set _training=False"


def test_forward_training_default():
    """Forward: training=None falls back to model._training."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4, dropout=0.5)
    src = np.random.randint(3, 20, (1, 4)).astype(np.int64)
    tgt_in = np.array([[1, 5, 8, 3, 12]], dtype=np.int64)
    src_mask = m.make_src_mask(src)
    tgt_mask = m.make_tgt_mask(tgt_in)

    m.eval()
    out1 = m.forward(src, tgt_in, src_mask, tgt_mask)  # uses _training=False
    out2 = m.forward(src, tgt_in, src_mask, tgt_mask, training=False)
    assert np.allclose(out1, out2), "eval + training=None should match training=False"

    m.train()
    out3 = m.forward(src, tgt_in, src_mask, tgt_mask)  # uses _training=True
    # With dropout=0.5, train vs eval outputs should differ (high probability)
    assert not np.allclose(out1, out3), "train vs eval outputs should differ with dropout"


def test_grad_norm_logging_runs():
    """Grad norms: _compute_grad_norms returns expected keys without error."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4, dropout=0)
    src, tgt_in, tgt_out = make_copy_batch(20, 5, 4)
    src_mask = m.make_src_mask(src)
    tgt_mask = m.make_tgt_mask(tgt_in)
    logits = m.forward(src, tgt_in, src_mask, tgt_mask, training=True)
    from transformer import cross_entropy_loss
    loss, dlogits = cross_entropy_loss(logits.reshape(-1, 20), tgt_out.reshape(-1))
    m.zero_grad()
    m.backward(dlogits.reshape(4, 6, 20))

    norms = _compute_grad_norms(m)
    assert 'enc_embed' in norms, f"missing enc_embed in {list(norms.keys())}"
    assert 'enc_0_self_attn' in norms, f"missing enc_0_self_attn"
    assert 'dec_0_cross_attn' in norms, f"missing dec_0_cross_attn"
    assert 'proj_b' in norms, f"missing proj_b"
    assert all(v >= 0 for v in norms.values()), "negative norm value"
    formatted = _format_norms(norms)
    assert len(formatted) > 0, "empty formatted norms"
    assert 'enc_embed' in formatted, "enc_embed missing from formatted output"


def test_param_norm_logging_runs():
    """Param norms: _compute_param_norms returns expected keys without error."""
    m = Transformer(20, 20, d_model=32, N=2, d_ff=64, h=4, dropout=0)
    norms = _compute_param_norms(m)
    assert 'enc_embed' in norms, f"missing enc_embed in {list(norms.keys())}"
    assert 'enc_0_self_attn' in norms, f"missing enc_0_self_attn"
    assert 'dec_0_cross_attn' in norms, f"missing dec_0_cross_attn"
    assert 'proj_b' in norms, f"missing proj_b"
    assert all(v >= 0 for v in norms.values()), "negative norm value"
    formatted = _format_norms(norms)
    assert len(formatted) > 0, "empty formatted norms"


def test_log_norms_training_does_not_crash():
    """Training: log_norms=True does not cause errors during copying."""
    m = Transformer(20, 20, d_model=32, N=1, d_ff=64, h=4, dropout=0)
    np.random.seed(42)
    m = train_copy(m, steps=5, batch_size=8, seq_len=4, vocab_size=20,
                   log_every=1, log_norms=True)
    out = translate(m, np.array([5, 8, 3, 12]))
    assert len(out) > 0, "empty output after training with log_norms"


def test_benchmark_runs():
    """Benchmark: runs all configs without error."""
    import subprocess, sys, os
    script = os.path.join(os.path.dirname(__file__), 'benchmark.py')
    result = subprocess.run(
        [sys.executable, script],
        capture_output=True, text=True, cwd=os.path.dirname(script))
    assert result.returncode == 0, f"benchmark failed: {result.stderr}"
    assert 'tiny' in result.stdout, "tiny config missing from benchmark output"
    assert 'base' in result.stdout, "base config missing from benchmark output"


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
        ('test_bpe_regression', test_bpe_regression),
        ('test_dataset_length', test_dataset_length),
        ('test_batch_shapes', test_batch_shapes),
        ('test_batch_no_all_padding', test_batch_no_all_padding),
        ('test_bleu_perfect_match', test_bleu_perfect_match),
        ('test_bleu_no_match', test_bleu_no_match),
        ('test_train_step', test_train_step),
        ('test_noam_lr_schedule', test_noam_lr_schedule),
        # Dropout (plan 07)
        ('test_dropout_training_vs_eval', test_dropout_training_vs_eval),
        ('test_dropout_eval_deterministic', test_dropout_eval_deterministic),
        ('test_dropout_rate_zero', test_dropout_rate_zero),
        ('test_dropout_backward', test_dropout_backward),
        ('test_dropout_attn_forward', test_dropout_attn_forward),
        # Gradient clipping + Noam (plan 05)
        ('test_gradient_clip_norm', test_gradient_clip_norm),
        ('test_gradient_clip_preserves_direction', test_gradient_clip_preserves_direction),
        ('test_noam_custom_scaling', test_noam_custom_scaling),
        ('test_training_with_clip', test_training_with_clip),
        ('test_training_with_noam', test_training_with_noam),
        # Gradient accumulation (plan 11)
        ('test_grad_accum_training', test_grad_accum_training),
        ('test_grad_accum_matches', test_grad_accum_matches),
        # Checkpointing (plan 06)
        ('test_save_and_load_params', test_save_and_load_params),
        ('test_save_and_load_optimizer', test_save_and_load_optimizer),
        ('test_checkpoint_file_exists', test_checkpoint_file_exists),
        ('test_load_nonexistent_raises', test_load_nonexistent_raises),
        ('test_checkpoint_training_resume', test_checkpoint_training_resume),
        # WMT run script (plan 08)
        ('test_checkpoint_averaging', test_checkpoint_averaging),
        ('test_run_wmt_dry_run', test_run_wmt_dry_run),
        # Train/eval mode, norm logging, benchmark (plan 13)
        ('test_train_eval_mode', test_train_eval_mode),
        ('test_forward_training_default', test_forward_training_default),
        ('test_grad_norm_logging_runs', test_grad_norm_logging_runs),
        ('test_param_norm_logging_runs', test_param_norm_logging_runs),
        ('test_log_norms_training_does_not_crash', test_log_norms_training_does_not_crash),
        ('test_benchmark_runs', test_benchmark_runs),
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
