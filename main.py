"""
Main pipeline for preprocessing D&D spells and training the Spellforge transformer.
Uses word-level tokenization end to end.
"""

from src.preprocessing import preprocess_and_save
from src.word_tokenizer import build_word_vocabulary, encode_spell_words
from src.splitting import split_dataset
from src.lm_data import create_lm_batches
from src.model import SpellforgeTransformer
from src.simple_train import simple_train


def load_spells(file_path):
    """Load spell blocks from the forged text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    parts = content.split("<<< New Spell Forged >>>")
    return ["<<< New Spell Forged >>>" + part for part in parts[1:]]


def encode_spells_words(spells, word_to_id):
    """Encode multiple spells using word-level tokenization."""
    return [encode_spell_words(spell, word_to_id) for spell in spells]


def main():
    print("Step 1: Preprocessing spells...")
    num_spells = preprocess_and_save("dnd-spells.csv", "spells.txt")
    print(f"Preprocessed {num_spells} spells and saved to spells.txt\n")

    print("Step 2: Loading spells...")
    spells = load_spells("spells.txt")
    print(f"Loaded {len(spells)} spells\n")

    print("Step 3: Splitting dataset...")
    train_spells, val_spells, test_spells = split_dataset(
        spells, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )
    print(f"Train spells: {len(train_spells)}")
    print(f"Val spells: {len(val_spells)}")
    print(f"Test spells: {len(test_spells)}\n")

    print("Step 4: Building + saving word vocabulary (train split only)...")
    word_to_id, id_to_word = build_word_vocabulary(train_spells, "tokenizer_vocab.json")
    print(f"Vocabulary size: {len(word_to_id)} unique tokens")
    print("Saved vocabulary to tokenizer_vocab.json\n")

    print("Step 5: Encoding train/val/test spells...")
    train_encoded = encode_spells_words(train_spells, word_to_id)
    val_encoded = encode_spells_words(val_spells, word_to_id)
    test_encoded = encode_spells_words(test_spells, word_to_id)
    print(f"Encoded train spells: {len(train_encoded)}")
    print(f"Encoded val spells: {len(val_encoded)}")
    print(f"Encoded test spells: {len(test_encoded)}\n")

    train_lengths = [len(seq) for seq in train_encoded]
    observed_max_seq_len = max(train_lengths)
    max_seq_len = 512
    print(
        f"Train sequence lengths - Min: {min(train_lengths)}, Max: {observed_max_seq_len}, "
        f"Avg: {sum(train_lengths) // len(train_lengths)}\n"
    )

    print("Step 6: Creating language-model batches...")
    batch_size = 4
    pad_id = word_to_id["<PAD>"]

    train_batches = create_lm_batches(
        train_encoded,
        batch_size,
        pad_token_id=pad_id,
        sort_by_length=True,
        shuffle=True,
    )
    val_batches = create_lm_batches(
        val_encoded,
        batch_size,
        pad_token_id=pad_id,
        sort_by_length=True,
        shuffle=False,
    )

    print(f"Train batches: {len(train_batches)}")
    print(f"Val batches: {len(val_batches)}")

    first_input_ids, first_target_ids = train_batches[0]
    print(f"First train input_ids shape: {tuple(first_input_ids.shape)}")
    print(f"First train target_ids shape: {tuple(first_target_ids.shape)}\n")

    print("Step 7: Creating model...")
    model = SpellforgeTransformer(
        vocab_size=len(word_to_id),
        max_seq_len=max_seq_len,
        d_model=384,
        num_layers=8,
        num_heads=6,
        d_ff=1536,
        dropout=0.1,
        pad_token_id=pad_id,
    )
    print(f"Model created with {model.get_num_params():,} parameters\n")

    print("Step 8: Training model...")
    total_tokens = sum(len(spell) for spell in train_encoded)
    eos_id = word_to_id["<EOS>"]
    eos_tokens = sum(spell.count(eos_id) for spell in train_encoded)
    eos_freq = eos_tokens / total_tokens
    eos_weight = 1.0

    print(f"EOS Analysis: {eos_tokens}/{total_tokens:,} tokens ({eos_freq:.4f})")
    print(f"Using EOS weight: {eos_weight:.1f}x\n")

    simple_train(
        model=model,
        train_batches=train_batches,
        val_batches=val_batches,
        token_to_id=word_to_id,
        id_to_token=id_to_word,
        epochs=40,
        learning_rate=1.0e-4,
        device="auto",
        eos_weight=eos_weight,
    )

    print("\nTraining pipeline complete!")


if __name__ == "__main__":
    main()
