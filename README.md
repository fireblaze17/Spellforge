# Spellforge

A D&D spell preprocessing and tokenization pipeline built with character-level encoding.

## Project Overview

Spellforge processes D&D spell data from CSV format, preprocesses the text, and builds a character-level tokenizer vocabulary for machine learning applications.

## Current Progress

**Implemented:**
- Spell preprocessing pipeline that reads from CSV and saves to text format
- Character-level vocabulary builder with character-to-ID and ID-to-character mappings
- Spell loading and encoding system
- Main pipeline orchestration with 4-step process:
  1. Preprocess spells from CSV to text file
  2. Build vocabulary from preprocessed text
  3. Load spells and encode using vocabulary
  4. Display sample encoded spells

## Project Structure

```
Spellforge/
├── main.py                 # Main pipeline entry point
├── src/
│   ├── preprocessing.py    # Spell preprocessing logic
│   ├── tokenizer.py        # Vocabulary building and encoding
│   └── __init__.py
├── dnd-spells.csv         # Input spell data
├── spells.txt             # Generated preprocessed spells
└── README.md              # This file
```

## How to Run

```bash
python main.py
```

This will:
- Process spells from the CSV file
- Generate vocabulary mappings
- Encode all spells using character-level tokens
- Display a sample encoded spell

## Technical Details

- Character-level tokenization approach
- Vocabulary built from all unique characters in preprocessed spells
- Spells encoded as sequences of character IDs

---

Last updated: February 5, 2026
