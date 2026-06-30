"""
Data pipeline for WMT training — Section 5.1 / 5.2

BPE tokenization (Section 5.1), bucketed batching, and BLEU evaluation.
Pure Python/NumPy — no external tokenization library needed.
"""
import os
import re
import json
from collections import Counter
import numpy as np
import sacrebleu


# ═══════════════════════════════════════════════════════════════════════
#  BPE Tokenizer (Section 5.1)
# ═══════════════════════════════════════════════════════════════════════

# Special token IDs
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SPECIAL_TOKENS = ['<pad>', '<unk>', '<s>', '</s>']


class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer (Section 5.1).

    Trains a BPE vocabulary from parallel text and provides encode/decode.
    Pure Python implementation — no external dependencies.

    The paper uses BPE with ~37,000 shared source/target tokens for EN-DE.
    """

    def __init__(self, model_path=None):
        self.vocab_size = 0
        self.id_to_token = {}       # int -> str (includes special tokens)
        self.token_to_id = {}       # str -> int
        self.bpe_merges = []        # list of (a, b) merge rules in order
        self._word_to_bpe_cache = {}  # cache for word -> BPE encoding

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def _get_char_vocab(self, texts):
        """Get initial character-level vocabulary from text."""
        chars = set()
        for t in texts:
            chars.update(t.split())
        # Add space token explicitly
        chars.add(' ')
        return sorted(chars)

    def train(self, src_path, tgt_path, model_prefix, vocab_size=8000):
        """
        Train a shared BPE model on concatenated source + target text.

        Args:
            src_path: path to source language training file
            tgt_path: path to target language training file
            model_prefix: output path prefix for .model and .vocab files
            vocab_size: total vocabulary size (including special tokens)
        """
        # Read all text
        with open(src_path, 'r', encoding='utf-8') as f:
            src_text = f.read().strip()
        with open(tgt_path, 'r', encoding='utf-8') as f:
            tgt_text = f.read().strip()

        combined = src_text + '\n' + tgt_text
        self._train_from_text(combined, vocab_size)
        self.save(model_prefix)

    def _train_from_text(self, text, vocab_size):
        """
        Train BPE from raw text string.

        Algorithm:
        1. Start with character vocabulary
        2. Count adjacent symbol pairs
        3. Merge the most frequent pair
        4. Repeat until vocab_size is reached
        """
        # Preprocess: split into words with end-of-word marker
        words = text.strip().split()
        # Count word frequencies
        word_freqs = Counter(words)

        # Start with character-level tokenization of each word
        # Each word is represented as a tuple of symbols (chars) + </w>
        self._word_to_bpe = {}
        char_vocab = set()
        for word in word_freqs:
            symbols = list(word) + ['</w>']
            self._word_to_bpe[word] = symbols
            for s in symbols:
                char_vocab.add(s)

        # Initial token set: all characters + </w>
        initial_tokens = sorted(char_vocab)

        # Map initial tokens to IDs (special tokens first)
        self.id_to_token = {i: t for i, t in enumerate(SPECIAL_TOKENS)}
        self.token_to_id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        next_id = len(SPECIAL_TOKENS)
        for t in initial_tokens:
            if t not in self.token_to_id:
                self.token_to_id[t] = next_id
                self.id_to_token[next_id] = t
                next_id += 1

        self.bpe_merges = []
        current_vocab_size = len(self.token_to_id)

        while current_vocab_size < vocab_size:
            # Count all adjacent pairs
            pair_counts = Counter()
            for word, freq in word_freqs.items():
                symbols = self._word_to_bpe[word]
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)

            # Merge: create new token from the pair
            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.token_to_id:
                self.token_to_id[merged_token] = next_id
                self.id_to_token[next_id] = merged_token
                next_id += 1

            self.bpe_merges.append(best_pair)

            # Apply merge to all words
            for word in self._word_to_bpe:
                symbols = self._word_to_bpe[word]
                new_symbols = []
                i = 0
                while i < len(symbols):
                    if (i < len(symbols) - 1
                            and symbols[i] == best_pair[0]
                            and symbols[i + 1] == best_pair[1]):
                        new_symbols.append(merged_token)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                self._word_to_bpe[word] = new_symbols

            current_vocab_size = next_id

        self.vocab_size = len(self.id_to_token)
        # Clear the word->bpe cache
        self._word_to_bpe_cache = {}

    def _bpe_encode_word(self, word):
        """Apply BPE merges to a single word."""
        if word in self._word_to_bpe_cache:
            return self._word_to_bpe_cache[word]

        symbols = list(word) + ['</w>']
        # Apply all merges in order
        changed = True
        while changed:
            changed = False
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                if pair in self._seen_merges:
                    symbols = symbols[:i] + [pair[0] + pair[1]] + symbols[i + 2:]
                    changed = True
                    break

        self._word_to_bpe_cache[word] = symbols
        return symbols

    def encode(self, text):
        """Tokenize text to integer IDs."""
        if not self.token_to_id:
            raise RuntimeError("BPETokenizer not trained or loaded")

        # Build a set of valid merge pairs for fast lookup
        self._seen_merges = set(self.bpe_merges)

        words = re.findall(r'\S+|\s+', text)
        ids = []
        for word in words:
            if word.strip():
                # BPE-encode the word
                bpe_symbols = self._bpe_encode_word(word.strip())
                for sym in bpe_symbols:
                    if sym in self.token_to_id:
                        ids.append(self.token_to_id[sym])
                    else:
                        ids.append(UNK_ID)

        return np.array(ids, dtype=np.int64)

    def decode(self, ids):
        """Convert integer IDs back to text."""
        tokens = []
        for i in ids:
            if i in self.id_to_token:
                tok = self.id_to_token[i]
                tokens.append(tok)
            else:
                tokens.append('<unk>')

        # Join tokens, remove </w> markers (they indicate end of word)
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def encode_sentence(self, sentence, bos=True, eos=True):
        """Encode a single sentence with optional BOS/EOS tokens."""
        ids = self.encode(sentence)
        if bos:
            ids = np.concatenate([[BOS_ID], ids])
        if eos:
            ids = np.concatenate([ids, [EOS_ID]])
        return ids

    def save(self, model_prefix):
        """Save model to files."""
        # Save merges
        with open(model_prefix + '.merges', 'w', encoding='utf-8') as f:
            for a, b in self.bpe_merges:
                f.write(f"{a} {b}\n")

        # Save vocabulary
        with open(model_prefix + '.vocab', 'w', encoding='utf-8') as f:
            for idx in sorted(self.id_to_token.keys()):
                f.write(f"{self.id_to_token[idx]} {idx}\n")

    def load(self, model_prefix):
        """Load a trained model."""
        # Load merges
        self.bpe_merges = []
        with open(model_prefix + '.merges', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 2:
                        self.bpe_merges.append((parts[0], parts[1]))

        # Load vocabulary
        self.id_to_token = {}
        self.token_to_id = {}
        with open(model_prefix + '.vocab', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.rsplit(' ', 1)
                    if len(parts) == 2:
                        token, idx = parts[0], int(parts[1])
                        self.id_to_token[idx] = token
                        self.token_to_id[token] = idx

        self.vocab_size = len(self.id_to_token)
        self._word_to_bpe_cache = {}


# ═══════════════════════════════════════════════════════════════════════
#  Parallel Dataset with Bucketed Batching
# ═══════════════════════════════════════════════════════════════════════

class ParallelDataset:
    """
    Load parallel sentences, BPE-tokenize, bucket by source length,
    and yield batches padded to the longest sequence in each bucket.
    """

    def __init__(self, src_path, tgt_path, tokenizer, max_len=100):
        """
        Args:
            src_path: source language file (one sentence per line)
            tgt_path: target language file (parallel, same line count)
            tokenizer: BPETokenizer instance
            max_len: max sequence length; longer sentences are skipped
        """
        self.tokenizer = tokenizer
        self.max_len = max_len

        print(f"Loading parallel data from:\n  src={src_path}\n  tgt={tgt_path}")
        with open(src_path, 'r', encoding='utf-8') as f:
            src_lines = f.readlines()
        with open(tgt_path, 'r', encoding='utf-8') as f:
            tgt_lines = f.readlines()

        assert len(src_lines) == len(tgt_lines), \
            f"src/tgt line count mismatch: {len(src_lines)} vs {len(tgt_lines)}"

        # Tokenize and filter by length
        self.src_ids = []
        self.tgt_ids = []
        self.src_texts = []
        self.tgt_texts = []
        skipped = 0
        for s, t in zip(src_lines, tgt_lines):
            s = s.strip()
            t = t.strip()
            if not s or not t:
                skipped += 1
                continue
            s_ids = tokenizer.encode_sentence(s, bos=True, eos=True)
            t_ids = tokenizer.encode_sentence(t, bos=True, eos=True)
            if len(s_ids) > max_len or len(t_ids) > max_len:
                skipped += 1
                continue
            self.src_ids.append(s_ids)
            self.tgt_ids.append(t_ids)
            self.src_texts.append(s)
            self.tgt_texts.append(t)

        self.n_pairs = len(self.src_ids)
        print(f"  {self.n_pairs} sentence pairs loaded"
              f" ({skipped} skipped due to length/filtering)")

        # Sort by src length for bucketing
        self._sorted_indices = sorted(
            range(self.n_pairs), key=lambda i: len(self.src_ids[i]))

    def batches(self, batch_size=32, shuffle=True):
        """
        Yield (src, tgt_in, tgt_out) batches.
        Each batch is padded to the max length within that bucket.

        Args:
            batch_size: number of sentence pairs per batch
            shuffle: shuffle batch order (but sentences stay length-sorted)

        Yields:
            src: (B, L_src) int64 array
            tgt_in: (B, L_tgt) int64 array (BOS-prefixed)
            tgt_out: (B, L_tgt) int64 array (EOS-suffixed)
        """
        indices = self._sorted_indices[:]
        if shuffle:
            # Shuffle within ~3x batch-size windows for randomness
            window = batch_size * 3
            for start in range(0, len(indices), window):
                chunk = indices[start:start + window]
                np.random.shuffle(chunk)
                indices[start:start + window] = chunk

        for b_start in range(0, len(indices), batch_size):
            batch_idx = indices[b_start:b_start + batch_size]
            B = len(batch_idx)

            # Get token sequences for this batch
            src_seqs = [self.src_ids[i] for i in batch_idx]
            tgt_seqs = [self.tgt_ids[i] for i in batch_idx]

            # Pad to max lengths in this batch
            L_src = max(len(s) for s in src_seqs)
            L_tgt = max(len(t) for t in tgt_seqs)

            src = np.zeros((B, L_src), dtype=np.int64)
            tgt_in = np.zeros((B, L_tgt), dtype=np.int64)
            tgt_out = np.zeros((B, L_tgt), dtype=np.int64)

            for i in range(B):
                s = src_seqs[i]
                t = tgt_seqs[i]
                src[i, :len(s)] = s
                # tgt_in: BOS + tokens (drop EOS)
                tgt_in[i, :len(t) - 1] = t[:-1]
                # tgt_out: tokens + EOS (drop BOS)
                tgt_out[i, :len(t) - 1] = t[1:]

            yield src, tgt_in, tgt_out


# ═══════════════════════════════════════════════════════════════════════
#  BLEU Scoring
# ═══════════════════════════════════════════════════════════════════════

def corpus_bleu(hypotheses, references):
    """
    Compute corpus-level BLEU using sacrebleu.

    Args:
        hypotheses: list of translated strings
        references: list of reference strings (single reference)

    Returns:
        dict with 'score', 'precision', 'bp', 'ratio', etc.
    """
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return {
        'score': bleu.score,
        'precision': bleu.precisions,
        'bp': bleu.bp,
        'ratio': bleu.sys_len / bleu.ref_len,
        'hyp_len': bleu.sys_len,
        'ref_len': bleu.ref_len,
    }


def sentence_bleu(hypothesis, reference):
    """Sentence-level BLEU (0-100)."""
    return sacrebleu.sentence_bleu(hypothesis, [reference]).score


# ═══════════════════════════════════════════════════════════════════════
#  Demo
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from transformer import Transformer, translate, train_wmt

    np.random.seed(42)

    print("=" * 50)
    print("Data Pipeline Demo")
    print("=" * 50)

    # ── Train BPE on tiny parallel corpus ──
    print("\n1. Training BPE tokenizer...")
    tok = BPETokenizer()
    tok.train('test_data/train.en', 'test_data/train.de',
              model_prefix='test_data/bpe_test', vocab_size=200)
    print(f"   Vocabulary size: {tok.vocab_size}")

    # Test encode/decode
    test_sentence = "hello world"
    ids = tok.encode(test_sentence)
    decoded = tok.decode(ids)
    print(f"   Encode/decode '{test_sentence}': {ids.tolist()} -> '{decoded}'")

    # ── Load dataset ──
    print("\n2. Loading parallel dataset...")
    dataset = ParallelDataset('test_data/train.en', 'test_data/train.de',
                              tok, max_len=50)
    print(f"   {dataset.n_pairs} sentence pairs")

    # ── Create batches ──
    print("\n3. Batching:")
    for i, (src, tgt_in, tgt_out) in enumerate(dataset.batches(batch_size=4)):
        print(f"   Batch {i}: src={src.shape}, tgt_in={tgt_in.shape}")
        if i >= 2:
            break

    # ── BLEU ──
    print("\n4. BLEU scoring:")
    result = corpus_bleu(["the cat sat on the mat"], ["the cat sat on the mat"])
    print(f"   Perfect match BLEU: {result['score']:.1f}")
    result = corpus_bleu(["goodbye world"], ["the cat sat on the mat"])
    print(f"   No match BLEU: {result['score']:.1f}")

    # ── Mini training demo ──
    print("\n5. Mini training demo:")
    dev_dataset = ParallelDataset('test_data/dev.en', 'test_data/dev.de',
                                  tok, max_len=50)
    model = Transformer(tok.vocab_size, tok.vocab_size,
                        d_model=32, N=2, d_ff=64, h=4)
    print(f"   Model params: {model.param_count():,}")
    model = train_wmt(model, dataset, steps=20, batch_size=4, d_model=32,
                       warmup_steps=10, log_every=5, eval_every=999)

    # ── Translate a sentence ──
    print("\n6. Translate demo (20 steps — expected to be garbage):")
    src_text = "the cat sat on the mat"
    src_ids = tok.encode_sentence(src_text, bos=True, eos=True)
    out_ids = translate(model, src_ids, bos_idx=BOS_ID, eos_idx=EOS_ID)
    out_text = tok.decode(out_ids)
    print(f"   src: {src_text}")
    print(f"   out: {out_text}")

    print("\nOK: data pipeline works!")
