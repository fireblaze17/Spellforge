import json
import re
from collections import Counter
from pathlib import Path


STRUCTURE_TOKENS = [
    "<<< New Recipe Forged >>>",
    "<<< May it Feed You Well >>>",
    "Name:",
    "Ingredients:",
    "Steps:",
    "Description:",
]

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
PUNCT_NO_LEADING_SPACE = {".", ",", ";", ":", "!", "?", ")", "]", "}", "%"}
PUNCT_NO_TRAILING_SPACE = {"(", "[", "{", "$"}
END_OF_WORD = "</w>"
CONTINUATION_PREFIX = "##"
DEFAULT_BPE_MERGES = 2000
MIN_PAIR_FREQUENCY = 2
MERGE_RANKS_CACHE = {}

TOKEN_PATTERN = re.compile(
    "|".join(re.escape(token) for token in STRUCTURE_TOKENS)
    + r"|[A-Za-z]+(?:['’][A-Za-z]+)*"
    + r"|\d+(?:[./-]\d+)*"
    + r"|[^\w\s]",
    re.UNICODE,
)


def tokenize_text(text):
    """
    Tokenize text while preserving recipe structure markers and field labels.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return TOKEN_PATTERN.findall(normalized)


def _is_bpe_candidate(token):
    """
    Only normal words/numbers should go through BPE.
    Structure markers and punctuation stay atomic.
    """
    if token in STRUCTURE_TOKENS:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*", token))


def _word_to_symbols(word):
    """
    Convert a word into initial character symbols with an end-of-word marker.
    """
    if word == "":
        return []

    symbols = list(word)
    symbols[-1] = symbols[-1] + END_OF_WORD
    return symbols


def _get_pair_stats(word_symbols, word_freqs):
    """
    Count adjacent symbol pairs weighted by word frequency.
    """
    pair_counts = Counter()
    for word, symbols in word_symbols.items():
        freq = word_freqs[word]
        for i in range(len(symbols) - 1):
            pair_counts[(symbols[i], symbols[i + 1])] += freq
    return pair_counts


def _merge_pair_in_symbols(symbols, pair):
    """
    Merge one symbol pair left-to-right inside a symbol sequence.
    """
    merged = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return merged


def _train_bpe_merges(word_freqs, num_merges=DEFAULT_BPE_MERGES, min_pair_frequency=MIN_PAIR_FREQUENCY):
    """
    Learn BPE merges from the word frequency table.
    """
    word_symbols = {word: _word_to_symbols(word) for word in word_freqs}
    merges = []

    for _ in range(num_merges):
        pair_counts = _get_pair_stats(word_symbols, word_freqs)
        if not pair_counts:
            break

        best_pair, best_count = pair_counts.most_common(1)[0]
        if best_count < min_pair_frequency:
            break

        merges.append(best_pair)
        for word, symbols in word_symbols.items():
            if len(symbols) >= 2:
                word_symbols[word] = _merge_pair_in_symbols(symbols, best_pair)

    return merges


def _apply_bpe_to_word(word, merge_ranks):
    """
    Apply learned BPE merges to one token and return display-friendly subword pieces.
    """
    symbols = _word_to_symbols(word)
    if not symbols:
        return []

    while len(symbols) > 1:
        candidate_pairs = [
            (merge_ranks[(symbols[i], symbols[i + 1])], (symbols[i], symbols[i + 1]))
            for i in range(len(symbols) - 1)
            if (symbols[i], symbols[i + 1]) in merge_ranks
        ]
        if not candidate_pairs:
            break

        _, best_pair = min(candidate_pairs, key=lambda item: item[0])
        symbols = _merge_pair_in_symbols(symbols, best_pair)

    pieces = []
    for i, symbol in enumerate(symbols):
        piece = symbol
        if piece.endswith(END_OF_WORD):
            piece = piece[: -len(END_OF_WORD)]

        if i > 0:
            piece = CONTINUATION_PREFIX + piece
        pieces.append(piece)

    return pieces


def _encode_text_tokens(text, merge_ranks):
    """
    Encode raw text into atomic structure/punctuation tokens plus BPE pieces.
    """
    encoded_tokens = []
    for token in tokenize_text(text):
        if _is_bpe_candidate(token):
            encoded_tokens.extend(_apply_bpe_to_word(token, merge_ranks))
        else:
            encoded_tokens.append(token)
    return encoded_tokens


def _store_merge_ranks(token_to_id, merge_ranks):
    """
    Attach merge-rank metadata to a tokenizer mapping without polluting the visible vocab.
    """
    MERGE_RANKS_CACHE[id(token_to_id)] = merge_ranks


def _get_merge_ranks(token_to_id):
    """
    Retrieve merge-rank metadata for a tokenizer mapping.
    """
    merge_ranks = MERGE_RANKS_CACHE.get(id(token_to_id))
    if merge_ranks is None:
        raise ValueError("tokenizer mapping is missing BPE merge data")
    return merge_ranks


def build_word_vocabulary(records, save_path="tokenizer_vocab.json", num_merges=DEFAULT_BPE_MERGES):
    """
    Build a structure-aware BPE vocabulary from training recipes.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    word_freqs = Counter()
    for record in records:
        for token in tokenize_text(record):
            if _is_bpe_candidate(token):
                word_freqs[token] += 1

    merges = _train_bpe_merges(word_freqs, num_merges=num_merges)
    merge_ranks = {pair: idx for idx, pair in enumerate(merges)}

    token_counts = Counter()
    for record in records:
        token_counts.update(_encode_text_tokens(record, merge_ranks))

    sorted_tokens = [token for token, _ in token_counts.most_common()]
    vocab = SPECIAL_TOKENS + sorted_tokens

    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    id_to_token = {idx: token for idx, token in enumerate(vocab)}

    vocab_data = {
        "wordToId": token_to_id,
        "idToWord": id_to_token,
        "vocab_size": len(vocab),
        "special_tokens": SPECIAL_TOKENS,
        "structure_tokens": STRUCTURE_TOKENS,
        "bpe_merges": [list(pair) for pair in merges],
        "num_merges": len(merges),
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)

    _store_merge_ranks(token_to_id, merge_ranks)

    print(f"BPE vocabulary built: {len(vocab)} tokens")
    print(f"Learned BPE merges: {len(merges)}")
    print(f"Most common tokens: {sorted_tokens[:20]}")

    return token_to_id, id_to_token


def build_recipe_vocabulary(records, save_path="tokenizer_vocab.json", num_merges=DEFAULT_BPE_MERGES):
    """Preferred recipe-specific vocabulary builder."""
    return build_word_vocabulary(records, save_path=save_path, num_merges=num_merges)


def encode_recipe_words(recipe, word_to_id):
    """
    Encode a recipe string into token IDs using structure-aware BPE tokenization.
    """
    token_ids = [word_to_id["<BOS>"]]

    merges = _get_merge_ranks(word_to_id)

    for token in _encode_text_tokens(recipe, merges):
        token_ids.append(word_to_id.get(token, word_to_id["<UNK>"]))

    token_ids.append(word_to_id["<EOS>"])
    return token_ids


def _append_token_text(parts, token):
    if not parts:
        parts.append(token)
        return

    prev = parts[-1]
    if token in PUNCT_NO_LEADING_SPACE:
        parts[-1] = prev.rstrip() + token
    elif prev and prev[-1] in PUNCT_NO_TRAILING_SPACE:
        parts[-1] = prev + token
    else:
        parts.append(" " + token)


def _reconstruct_tokens(tokens):
    """
    Convert BPE pieces back into normal text tokens.
    """
    reconstructed = []
    current_word = ""

    def flush_word():
        nonlocal current_word
        if current_word != "":
            reconstructed.append(current_word)
            current_word = ""

    for token in tokens:
        if token in STRUCTURE_TOKENS:
            flush_word()
            reconstructed.append(token)
        elif token.startswith(CONTINUATION_PREFIX):
            current_word += token[len(CONTINUATION_PREFIX):]
        elif _is_bpe_candidate(token):
            flush_word()
            current_word = token
        else:
            flush_word()
            reconstructed.append(token)

    flush_word()
    return reconstructed


def decode_tokens_words(token_ids, id_to_word):
    """
    Decode token IDs back into canonical recipe text.
    """
    raw_tokens = []
    for token_id in token_ids:
        token = id_to_word.get(token_id)
        if token is None or token in SPECIAL_TOKENS:
            continue
        raw_tokens.append(token)

    tokens = _reconstruct_tokens(raw_tokens)
    lines = []
    current_parts = []
    current_section = None

    def flush_line():
        if current_parts:
            lines.append("".join(current_parts).rstrip())
            current_parts.clear()

    for token in tokens:
        if token == "<<< New Recipe Forged >>>":
            flush_line()
            current_section = None
            lines.append(token)
        elif token == "<<< May it Feed You Well >>>":
            flush_line()
            current_section = None
            lines.append(token)
        elif token in {"Name:", "Description:"}:
            flush_line()
            current_section = token
            current_parts.append("  " + token)
        elif token in {"Ingredients:", "Steps:"}:
            flush_line()
            current_section = token
            current_parts.append("  " + token)
        elif token == "-" and current_section in {"Ingredients:", "Steps:"}:
            flush_line()
            current_parts.append("  -")
        else:
            _append_token_text(current_parts, token)

    flush_line()
    return "\n".join(lines).strip()


def encode_text_words(text, word_to_id, add_bos=False, add_eos=False):
    """
    Encode arbitrary text with the same tokenizer used for recipes.
    """
    token_ids = []
    if add_bos:
        token_ids.append(word_to_id["<BOS>"])

    merges = _get_merge_ranks(word_to_id)

    for token in _encode_text_tokens(text, merges):
        token_ids.append(word_to_id.get(token, word_to_id["<UNK>"]))

    if add_eos:
        token_ids.append(word_to_id["<EOS>"])

    return token_ids


def load_word_vocabulary(vocab_path="tokenizer_vocab.json"):
    """
    Load BPE vocabulary from JSON file.
    """
    vocab_path = Path(vocab_path)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    token_to_id = vocab_data["wordToId"]
    id_to_token = {int(k): v for k, v in vocab_data["idToWord"].items()}
    merges = [tuple(pair) for pair in vocab_data.get("bpe_merges", [])]
    _store_merge_ranks(token_to_id, {pair: idx for idx, pair in enumerate(merges)})
    return token_to_id, id_to_token


def load_recipe_vocabulary(vocab_path="tokenizer_vocab.json"):
    """Preferred recipe-specific vocabulary loader."""
    return load_word_vocabulary(vocab_path)


def build_vocabulary(records, save_path="tokenizer_vocab.json"):
    """Compatibility wrapper that redirects to the BPE tokenizer."""
    return build_word_vocabulary(records, save_path)


def load_vocabulary(vocab_path="tokenizer_vocab.json"):
    """Compatibility wrapper that redirects to the BPE tokenizer."""
    return load_word_vocabulary(vocab_path)
