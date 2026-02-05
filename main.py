"""
Main pipeline for preprocessing D&D spells and building a character-level tokenizer.
"""

from src.preprocessing import preprocess_and_save
from src.tokenizer import build_vocabulary, load_spells, encode_spells

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
    print(f"Encoded {len(encoded_spells)} spells\n")
    
    # Step 4: Display sample
    print("Sample encoded spell:")
    print(encoded_spells[0])
    print()
    

if __name__ == "__main__":
    main()
