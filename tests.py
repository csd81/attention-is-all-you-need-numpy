"""
Unit tests for the Transformer implementation.
"""
import sys
import math
import numpy as np

sys.path.insert(0, '.')
from transformer import (
    Transformer, translate, translate_beam, train_copy,
    make_copy_batch, Param, softmax,
)


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
