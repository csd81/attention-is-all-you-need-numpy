"""
Attention Visualization — Section 4

Plot attention heatmaps for encoder self-attention, decoder self-attention,
and decoder cross-attention across all layers and heads.
"""
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _tokens_to_str(tokens):
    """Convert integer token IDs to readable labels."""
    return [str(t) if t > 2 else ['PAD', 'BOS', 'EOS'][t] for t in tokens]


def plot_attention_grid(attn_weights, tokens=None, title='Attention',
                        figsize=(16, 12), cmap='viridis', show_ticks=True):
    """
    Plot a grid of attention heatmaps: rows=layers, cols=heads.

    Args:
        attn_weights: list of (h, L, L) arrays, one per layer, or (N_layers, h, L, L)
        tokens: optional list of token labels
        title: figure title
        figsize: (width, height)
    Returns:
        matplotlib figure
    """
    attn = np.asarray(attn_weights)
    if attn.ndim == 4:
        N, h, L, _ = attn.shape
    elif attn.ndim == 3:
        N = 1
        h, L, _ = attn.shape
        attn = attn[None, :, :, :]
    else:
        raise ValueError(f'Expected 3D or 4D, got {attn.ndim}D')

    fig, axes = plt.subplots(N, h, figsize=figsize,
                             squeeze=False, constrained_layout=True)
    fig.suptitle(title, fontsize=14)

    for layer in range(N):
        for head in range(h):
            ax = axes[layer, head]
            im = ax.imshow(attn[layer, head], cmap=cmap, vmin=0, vmax=1)

            if show_ticks and tokens is not None:
                ax.set_xticks(range(L))
                ax.set_xticklabels(tokens, rotation=90, fontsize=6)
                ax.set_yticks(range(L))
                ax.set_yticklabels(tokens, fontsize=6)
            else:
                ax.tick_params(bottom=False, left=False,
                               labelbottom=False, labelleft=False)

            if layer == 0:
                ax.set_title(f'Head {head}', fontsize=9)
            if head == 0:
                ax.set_ylabel(f'Layer {layer}', fontsize=9)

    return fig


def plot_attention_head(attn, tokens=None, title='Attention Head',
                        figsize=(6, 5), cmap='viridis', show_colorbar=True):
    """
    Plot a single attention heatmap for one head.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    attn = np.asarray(attn)
    L = attn.shape[-1]

    im = ax.imshow(attn, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title)

    if tokens is not None:
        ax.set_xticks(range(L))
        ax.set_xticklabels(tokens, rotation=90, fontsize=8)
        ax.set_yticks(range(L))
        ax.set_yticklabels(tokens, fontsize=8)
    else:
        ax.set_xlabel('Keys')
        ax.set_ylabel('Queries')

    if show_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046)

    return fig


def visualize_all(model, src, tgt_in, src_tokens=None, tgt_tokens=None,
                  save_path=None):
    """
    Full visualization: run forward with attention capture and plot all maps.

    Args:
        model: trained Transformer
        src: (1, L_src) source token array
        tgt_in: (1, L_tgt) target input token array
        src_tokens, tgt_tokens: optional label lists
        save_path: if set, save figure to this path
    """
    src_mask = model.make_src_mask(src)
    tgt_mask = model.make_tgt_mask(tgt_in)

    logits, attn = model.forward_with_attention(src, tgt_in, src_mask, tgt_mask)

    src_len = src.shape[1]
    tgt_len = tgt_in.shape[1]

    figs = []

    # ── Encoder self-attention ──
    enc_self = [a[0] for a in attn['encoder']['self_attn']]  # (h, L, L) per layer
    fig = plot_attention_grid(enc_self, tokens=src_tokens,
                              title='Encoder Self-Attention')
    figs.append(('encoder_self_attention', fig))

    # ── Decoder self-attention ──
    dec_self = [a[0] for a in attn['decoder']['self_attn']]
    fig = plot_attention_grid(dec_self, tokens=tgt_tokens,
                              title='Decoder Self-Attention')
    figs.append(('decoder_self_attention', fig))

    # ── Decoder cross-attention ──
    dec_cross = [a[0] for a in attn['decoder']['cross_attn']]
    fig = plot_attention_grid(dec_cross, tokens=tgt_tokens,
                              title='Decoder Cross-Attention (→ Encoder)')
    figs.append(('decoder_cross_attention', fig))

    if save_path:
        for name, fig in figs:
            fig.savefig(f'{save_path}_{name}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

    return figs, logits, attn


def visualize_single_head(model, src, tgt_in, layer=0, head=0, attn_type='cross',
                          src_tokens=None, tgt_tokens=None, save_path=None):
    """
    Plot a single attention head from a specific layer.
    attn_type: 'enc_self', 'dec_self', or 'cross'
    """
    src_mask = model.make_src_mask(src)
    tgt_mask = model.make_tgt_mask(tgt_in)
    logits, attn = model.forward_with_attention(src, tgt_in, src_mask, tgt_mask)

    if attn_type == 'enc_self':
        w = attn['encoder']['self_attn'][layer][0, head]
        tokens = src_tokens
        title = f'Encoder Self-Attn  Layer {layer}  Head {head}'
    elif attn_type == 'dec_self':
        w = attn['decoder']['self_attn'][layer][0, head]
        tokens = tgt_tokens
        title = f'Decoder Self-Attn  Layer {layer}  Head {head}'
    else:
        w = attn['decoder']['cross_attn'][layer][0, head]
        tokens = tgt_tokens
        title = f'Cross-Attention  Layer {layer}  Head {head}'

    fig = plot_attention_head(w, tokens=tokens, title=title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, w


if __name__ == '__main__':
    # Demo: train a small model and visualize attention
    import sys
    sys.path.insert(0, '.')
    from transformer import Transformer, train_copy, translate, make_copy_batch

    np.random.seed(42)
    VOCAB = 20
    SEQ_LEN = 4

    model = Transformer(VOCAB, VOCAB, d_model=32, N=2, d_ff=64, h=4)
    model = train_copy(model, steps=200, batch_size=16, seq_len=SEQ_LEN,
                       vocab_size=VOCAB, log_every=200)

    # Run a test sample with attention capture
    src = np.array([[5, 8, 3, 12]], dtype=np.int64)
    tgt_in = np.array([[1, 5, 8, 3, 12]], dtype=np.int64)  # BOS + src

    src_tok = _tokens_to_str(src[0].tolist())
    tgt_tok = _tokens_to_str(tgt_in[0].tolist())

    print(f"\nGenerating attention plots for src={src[0].tolist()}...")
    figs, logits, _ = visualize_all(model, src, tgt_in, src_tok, tgt_tok,
                                    save_path='attn_demo')
    plt.close('all')
    print("Saved: attn_demo_encoder_self_attention.png")
    print("Saved: attn_demo_decoder_self_attention.png")
    print("Saved: attn_demo_decoder_cross_attention.png")
    print("OK: visualization works!")
