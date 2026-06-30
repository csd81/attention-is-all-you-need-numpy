"""
Training speed benchmark for the Transformer implementation.

Creates models of increasing size, runs forward+backward passes,
and reports time per step to give a rough sense of throughput.
"""
import time
import numpy as np
from transformer import Transformer, make_copy_batch, cross_entropy_loss


def benchmark_config(name, vocab, d_model, N, d_ff, h, steps=50, batch_size=16, seq_len=10):
    """Run steps train steps and report ms/step."""
    np.random.seed(42)
    model = Transformer(vocab, vocab, d_model=d_model, N=N, d_ff=d_ff, h=h, dropout=0)

    # Warmup
    src, tgt_in, tgt_out = make_copy_batch(vocab, seq_len, batch_size)
    src_mask = model.make_src_mask(src)
    tgt_mask = model.make_tgt_mask(tgt_in)
    logits = model.forward(src, tgt_in, src_mask, tgt_mask, training=True)
    B, L, C = logits.shape
    loss, dlogits = cross_entropy_loss(logits.reshape(-1, C), tgt_out.reshape(-1))
    model.zero_grad()
    model.backward(dlogits.reshape(B, L, C))

    # Timed run
    t0 = time.perf_counter()
    for _ in range(steps):
        src, tgt_in, tgt_out = make_copy_batch(vocab, seq_len, batch_size)
        src_mask = model.make_src_mask(src)
        tgt_mask = model.make_tgt_mask(tgt_in)
        logits = model.forward(src, tgt_in, src_mask, tgt_mask, training=True)
        B, L, C = logits.shape
        loss, dlogits = cross_entropy_loss(logits.reshape(-1, C), tgt_out.reshape(-1))
        model.zero_grad()
        model.backward(dlogits.reshape(B, L, C))

    elapsed = time.perf_counter() - t0
    ms_per_step = elapsed / steps * 1000
    params = model.param_count()
    return params, ms_per_step


def main():
    configs = [
        ("tiny",     20,    32, 2,  64, 4),
        ("small",    200,  128, 3, 256, 4),
        ("base",   37000,  512, 6, 2048, 8),
    ]

    print(f"{'model':>10}  {'params':>12}  {'ms/step':>10}")
    print("-" * 36)
    for name, vocab, d_model, N, d_ff, h in configs:
        steps = 20 if name == "base" else 50
        batch_size = 8 if name == "base" else 16
        params, ms = benchmark_config(name, vocab, d_model, N, d_ff, h,
                                       steps=steps, batch_size=batch_size)
        print(f"{name:>10}  {params:>12,}  {ms:>10.3f}")


if __name__ == '__main__':
    main()
