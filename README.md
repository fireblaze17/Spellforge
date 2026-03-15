# Spellforge

A D&D spell preprocessing, tokenization, and transformer training pipeline built around word-level tokens.

## Project Overview

Spellforge processes D&D spell data from CSV format, preprocesses it into a consistent spell template, builds a word-level vocabulary from the training split, and trains an autoregressive transformer to generate new spells.

## Current Progress

Implemented:
- Spell preprocessing from CSV to formatted text blocks
- Train/validation/test splitting before vocabulary construction to avoid leakage
- Word-level vocabulary builder with special tokens `<PAD>`, `<BOS>`, `<EOS>`, and `<UNK>`
- Word-level spell encoding for language-model training
- Transformer language model, training loop, and evaluation scripts

## Project Structure

```text
Spellforge/
|-- main.py
|-- src/
|   |-- preprocessing.py
|   |-- word_tokenizer.py
|   |-- tokenizer.py
|   |-- lm_data.py
|   |-- model.py
|   |-- simple_train.py
|   `-- ...
|-- dnd-spells.csv
|-- spells.txt
`-- tokenizer_vocab.json
```

`src/tokenizer.py` is now a compatibility layer that forwards to the word tokenizer.

## How to Run

```bash
python main.py
```

This will:
- preprocess spells from the CSV file
- split the dataset
- build a word vocabulary from the training split
- encode spells into token IDs
- create LM batches
- train the transformer

## Technical Details

- Word-level tokenization with punctuation kept as separate tokens
- Vocabulary built from training data only
- Spells encoded as token ID sequences with BOS/EOS boundaries
- Causal transformer trained for next-token prediction
