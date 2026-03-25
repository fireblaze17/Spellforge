# Spellforge

Spellforge is a recipe generator built on a custom decoder-only transformer. The active workflow preprocesses recipes from `RAW_recipes.csv`, builds a structure-aware BPE vocabulary from the training split only, trains an autoregressive model, and evaluates recipe-format generation quality.

## Repository Layout

```text
Spellforge/
|-- main.py
|-- preprocess_recipes.py
|-- evaluate_model.py
|-- data/
|   |-- raw/
|   `-- processed/
|-- artifacts/
|   |-- models/
|   `-- tokenizer/
`-- src/
    |-- config.py
    |-- preprocessing.py
    |-- word_tokenizer.py
    |-- lm_data.py
    |-- model.py
    |-- simple_train.py
    `-- evaluation.py
```

Legacy root-level files such as `RAW_recipes.csv`, `recipes.txt`, and `tokenizer_vocab.json` are still supported as fallbacks, but new outputs are written under `data/processed/` and `artifacts/`.

## Active Workflow

`preprocess_recipes.py`
- Preprocesses raw recipes into the canonical text format used for language-model training.

`main.py`
- Preprocesses recipes
- Splits train/validation/test
- Builds the BPE vocabulary from the training split only
- Encodes recipes into token IDs
- Trains the recipe model
- Evaluates on the held-out test split

`evaluate_model.py`
- Loads a trained checkpoint
- Generates sample recipes
- Reports format-compliance metrics

## Setup

Create a Python environment with the packages used by the repo:

```bash
pip install torch pandas tqdm
```

Place the dataset at `data/raw/RAW_recipes.csv`. If you still have it at the repo root as `RAW_recipes.csv`, the code will use that as a fallback.

## Commands

```bash
python preprocess_recipes.py
python main.py
python evaluate_model.py
```

## Current Model Design

- Structure-aware BPE tokenization with recipe boundary markers and field labels preserved
- Vocabulary built from the training split only
- Decoder-only transformer with causal masking
- AdamW optimization, cosine learning-rate decay, checkpointing, and sample generation during training
