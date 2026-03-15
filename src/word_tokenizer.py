import json
import re
from collections import Counter


STRUCTURE_TOKENS = [
    "<<< New Spell Forged >>>",
    "<<< May it Serve You Well >>>",
    "Name:",
    "Classes:",
    "School:",
    "Range:",
    "Duration:",
    "Description:",
]

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
PUNCT_NO_LEADING_SPACE = {".", ",", ";", ":", "!", "?", ")", "]", "}", "%"}
PUNCT_NO_TRAILING_SPACE = {"(", "[", "{", "$"}

TOKEN_PATTERN = re.compile(
    "|".join(re.escape(token) for token in STRUCTURE_TOKENS)
    + r"|[A-Za-z]+(?:['’][A-Za-z]+)*"
    + r"|\d+(?:st|nd|rd|th)?"
    + r"|[^\w\s]",
    re.UNICODE,
)


def build_word_vocabulary(spells, save_path="tokenizer_vocab.json"):
    """
    Build a structure-aware word vocabulary from training spells.
    """
    all_words = []
    for spell in spells:
        all_words.extend(tokenize_text(spell))

    word_counts = Counter(all_words)
    sorted_words = [word for word, _ in word_counts.most_common()]
    vocab = SPECIAL_TOKENS + sorted_words

    word_to_id = {word: idx for idx, word in enumerate(vocab)}
    id_to_word = {idx: word for idx, word in enumerate(vocab)}

    vocab_data = {
        "wordToId": word_to_id,
        "idToWord": id_to_word,
        "vocab_size": len(vocab),
        "special_tokens": SPECIAL_TOKENS,
        "structure_tokens": STRUCTURE_TOKENS,
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, indent=2, ensure_ascii=False)

    print(f"Word vocabulary built: {len(vocab)} words")
    print(f"Most common words: {sorted_words[:20]}")

    return word_to_id, id_to_word


def tokenize_text(text):
    """
    Tokenize text while preserving spell structure markers and field labels.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return TOKEN_PATTERN.findall(normalized)


def encode_spell_words(spell, word_to_id):
    """
    Encode a spell string into token IDs using structure-aware word tokenization.
    """
    tokens = [word_to_id["<BOS>"]]
    for word in tokenize_text(spell):
        tokens.append(word_to_id.get(word, word_to_id["<UNK>"]))
    tokens.append(word_to_id["<EOS>"])
    return tokens


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


def decode_tokens_words(token_ids, id_to_word):
    """
    Decode token IDs back into canonical spell text.
    """
    tokens = []
    for token_id in token_ids:
        word = id_to_word.get(token_id)
        if word is None or word in SPECIAL_TOKENS:
            continue
        tokens.append(word)

    lines = []
    current_parts = []

    def flush_line():
        if current_parts:
            lines.append("".join(current_parts).rstrip())
            current_parts.clear()

    for token in tokens:
        if token == "<<< New Spell Forged >>>":
            flush_line()
            lines.append(token)
        elif token == "<<< May it Serve You Well >>>":
            flush_line()
            lines.append(token)
        elif token in STRUCTURE_TOKENS[2:]:
            flush_line()
            current_parts.append("  " + token)
        else:
            _append_token_text(current_parts, token)

    flush_line()
    return "\n".join(lines).strip()


def encode_text_words(text, word_to_id, add_bos=False, add_eos=False):
    """
    Encode arbitrary text with the same tokenizer used for spells.
    """
    tokens = []
    if add_bos:
        tokens.append(word_to_id["<BOS>"])

    for word in tokenize_text(text):
        tokens.append(word_to_id.get(word, word_to_id["<UNK>"]))

    if add_eos:
        tokens.append(word_to_id["<EOS>"])

    return tokens


def load_word_vocabulary(vocab_path="tokenizer_vocab.json"):
    """
    Load word vocabulary from JSON file.
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    word_to_id = vocab_data["wordToId"]
    id_to_word = {int(k): v for k, v in vocab_data["idToWord"].items()}
    return word_to_id, id_to_word


def build_vocabulary(spells, save_path="tokenizer_vocab.json"):
    """Compatibility wrapper that redirects to the word-level tokenizer."""
    return build_word_vocabulary(spells, save_path)


def load_vocabulary(vocab_path="tokenizer_vocab.json"):
    """Compatibility wrapper that redirects to the word-level tokenizer."""
    return load_word_vocabulary(vocab_path)
