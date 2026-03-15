"""
Main pipeline for preprocessing D&D spells and building a character-level tokenizer.
"""

from src.preprocessing import preprocess_and_save
from src.tokenizer import build_vocabulary, load_spells, encode_spells, save_vocabulary
from src.splitting import split_dataset
from src.lm_data import create_lm_batches

def main():
    # Step 1: Preprocess spells and save to text file
    print("Step 1: Preprocessing spells...")
    num_spells = preprocess_and_save('dnd-spells.csv', 'spells.txt')
    print(f"Preprocessed {num_spells} spells and saved to spells.txt\n")

    # Step 2: Load spells
    print("Step 2: Loading spells...")
    spells = load_spells('spells.txt')
    print(f"Loaded {len(spells)} spells\n")

    # Step 3: Split raw spells so vocab can be built on train only (no leakage)
    print("Step 3: Splitting dataset...")
    train_spells, val_spells, test_spells = split_dataset(
        spells, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42
    )
    print(f"Train spells: {len(train_spells)}")
    print(f"Val spells: {len(val_spells)}")
    print(f"Test spells: {len(test_spells)}\n")

    # Step 4: Build vocabulary from train split only, then save it
    print("Step 4: Building + saving vocabulary (train split only)...")
    train_text = "".join(train_spells)
    charToId, idToChar = build_vocabulary(train_text)
    save_vocabulary(charToId, idToChar, "tokenizer_vocab.json")
    print(f"Vocabulary size: {len(charToId)} unique characters")
    print("Saved vocabulary to tokenizer_vocab.json\n")

    # Step 5: Encode each split with the same train-built vocabulary
    print("Step 5: Encoding train/val/test spells...")
    train_encoded = encode_spells(train_spells, charToId)
    val_encoded = encode_spells(val_spells, charToId)
    test_encoded = encode_spells(test_spells, charToId)
    print(f"Encoded train spells: {len(train_encoded)}")
    print(f"Encoded val spells: {len(val_encoded)}")
    print(f"Encoded test spells: {len(test_encoded)}\n")

    # train sequence stats (helpful for max length expectations)
    train_lengths = [len(seq) for seq in train_encoded]
    print(
        f"Train sequence lengths - Min: {min(train_lengths)}, Max: {max(train_lengths)}, "
        f"Avg: {sum(train_lengths) // len(train_lengths)}\n"
    )

    # Step 6: Create LM batches (input_ids and target_ids)
    print("Step 6: Creating language-model batches...")
    batch_size = 32
    pad_id = charToId["<PAD>"]

    train_batches = create_lm_batches(train_encoded, batch_size, pad_token_id=pad_id, sort_by_length=True, shuffle=True)
    val_batches = create_lm_batches(val_encoded, batch_size, pad_token_id=pad_id, sort_by_length=True, shuffle=False)
    test_batches = create_lm_batches(test_encoded, batch_size, pad_token_id=pad_id, sort_by_length=True, shuffle=False)

    print(f"Train batches: {len(train_batches)}")
    print(f"Val batches: {len(val_batches)}")
    print(f"Test batches: {len(test_batches)}")

    first_input_ids, first_target_ids = train_batches[0]
    print(f"First train input_ids shape: {tuple(first_input_ids.shape)}")
    print(f"First train target_ids shape: {tuple(first_target_ids.shape)}")
    print("Sample input_ids (first 2 rows, first 20 tokens):")
    print(first_input_ids[:2, :20])
    print("Sample target_ids (first 2 rows, first 20 tokens):")
    print(first_target_ids[:2, :20])
    print()


if __name__ == "__main__":
    main()
