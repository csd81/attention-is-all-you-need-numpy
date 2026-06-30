# Attention Is All You Need — NumPy Implementation

A from-scratch implementation of the Transformer architecture (Vaswani et al., 2017) using **only NumPy** — no PyTorch, no TensorFlow, no JAX.

Each layer implements manual `forward()` and `backward()` methods, making the full gradient flow explicit and educational.

## Architecture

| Component | Paper Section | File |
|---|---|---|
| Scaled Dot-Product Attention | 3.2.1 | `attention/transformer.py` |
| Multi-Head Attention | 3.2.2 | `attention/transformer.py` |
| Position-wise FFN | 3.3 | `attention/transformer.py` |
| Positional Encoding | 3.5 | `attention/transformer.py` |
| Encoder / Decoder stacks | 3.1 | `attention/transformer.py` |
| Full Transformer | 3 | `attention/transformer.py` |

## Features

- ✅ Pure NumPy, zero DL framework dependencies
- ✅ Manual backprop through every layer
- ✅ Adam optimizer with label smoothing
- ✅ Greedy decoding
- ✅ Beam search (beam size, length penalty α)
- ✅ Trains to 100% on sequence copying in ~1s

## Run

```bash
pip install numpy
python attention/transformer.py
```

## Plans

See [`attention/plan/`](attention/plan/) for high-ROI features to implement next:
- Weight tying (Section 3.4)
- Attention visualization
- WMT data training
- Noam LR schedule (Section 5.3)
- Checkpointing
- Dropout (Section 5.4)

## Paper

[Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., NIPS 2017
