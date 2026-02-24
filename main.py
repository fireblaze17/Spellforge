"""
Main pipeline for preprocessing D&D spells and building a character-level tokenizer.
"""

from src.preprocessing import preprocess_and_save
from src.tokenizer import build_vocabulary, load_spells, encode_spells
from src.batching import create_batches

def main():
    # Step 1: Preprocess spells and save to text file
    print("Step 1: Preprocessing spells...")
    num_spells = preprocess_and_save('dnd-spells.csv', 'spells.txt')
    print(f"Preprocessed {num_spells} spells and saved to spells.txt\n")
    
    # Step 2: Load text and build vocabulary
    print("Step 2: Building vocabulary...")
    with open('spells.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    charToId, idToChar = build_vocabulary(text)
    print(f"Vocabulary size: {len(charToId)} unique characters\n")
    
    # Step 3: Load spells and encode them
    print("Step 3: Loading and encoding spells...")
    spells = load_spells('spells.txt')
    encoded_spells = encode_spells(spells, charToId)
    print(f"Encoded {len(encoded_spells)} spells")
    
    # Sequence length stats
    lengths = [len(seq) for seq in encoded_spells]
    print(f"Sequence lengths - Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths) // len(lengths)}\n")
    
    # Step 4: Create batches
    print("Step 4: Creating batches...")
    batch_size = 32
    batches = create_batches(encoded_spells, batch_size, sort_by_length=True, shuffle=True)
    print(f"Created {len(batches)} batches with batch_size={batch_size}")
    print(f"First batch shape: {batches[0].shape}")
    print(f"Sample batch (first 2 sequences, first 20 tokens):")
    print(batches[0][:2, :20])
    print()


if __name__ == "__main__":
    main()
