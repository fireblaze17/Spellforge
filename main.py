"""Main recipe training pipeline."""

from src.config import RECIPE_DATA_CONFIG, TRAINING_CONFIG
from src.evaluation import RecipeEvaluator
from src.lm_data import create_recipe_lm_batches
from src.model import RecipeTransformer
from src.preprocessing import preprocess_and_save
from src.simple_train import simple_train
from src.splitting import split_dataset
from src.word_tokenizer import build_recipe_vocabulary, encode_recipe_words


def load_recipes(file_path):
    """Load recipe blocks from the forged text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    parts = content.split("<<< New Recipe Forged >>>")
    return ["<<< New Recipe Forged >>>" + part for part in parts[1:]]


def encode_recipes(recipes, token_to_id):
    """Encode multiple recipes using the recipe tokenizer."""
    return [encode_recipe_words(recipe, token_to_id) for recipe in recipes]


def filter_recipes_by_length(encoded_recipes, max_seq_len):
    """Keep only encoded recipes that fit within the model context window."""
    kept = [recipe for recipe in encoded_recipes if len(recipe) <= max_seq_len]
    dropped = len(encoded_recipes) - len(kept)
    return kept, dropped


def main():
    print("Step 1: Preprocessing recipes...")
    source_csv = RECIPE_DATA_CONFIG.resolved_raw_recipes_csv()
    processed_corpus = RECIPE_DATA_CONFIG.processed_recipes_txt
    num_recipes = preprocess_and_save(
        source_csv,
        processed_corpus,
        max_recipes=TRAINING_CONFIG.max_examples,
        random_seed=TRAINING_CONFIG.split_seed,
    )
    print(f"Preprocessed {num_recipes} recipes and saved to {processed_corpus}\n")

    print("Step 2: Loading recipes...")
    recipes = load_recipes(processed_corpus)
    print(f"Loaded {len(recipes)} recipes\n")

    print("Step 3: Splitting dataset...")
    train_recipes, val_recipes, test_recipes = split_dataset(
        recipes,
        train_ratio=TRAINING_CONFIG.train_ratio,
        val_ratio=TRAINING_CONFIG.val_ratio,
        test_ratio=TRAINING_CONFIG.test_ratio,
        seed=TRAINING_CONFIG.split_seed,
    )
    print(f"Train recipes: {len(train_recipes)}")
    print(f"Val recipes: {len(val_recipes)}")
    print(f"Test recipes: {len(test_recipes)}\n")

    print("Step 4: Building + saving BPE vocabulary (train split only)...")
    vocab_path = RECIPE_DATA_CONFIG.tokenizer_vocab_json
    token_to_id, id_to_token = build_recipe_vocabulary(train_recipes, vocab_path)
    print(f"Vocabulary size: {len(token_to_id)} unique tokens")
    print(f"Saved vocabulary to {vocab_path}\n")

    print("Step 5: Encoding train/val/test recipes...")
    train_encoded = encode_recipes(train_recipes, token_to_id)
    val_encoded = encode_recipes(val_recipes, token_to_id)
    test_encoded = encode_recipes(test_recipes, token_to_id)

    max_seq_len = TRAINING_CONFIG.max_seq_len
    train_encoded, train_dropped = filter_recipes_by_length(train_encoded, max_seq_len)
    val_encoded, val_dropped = filter_recipes_by_length(val_encoded, max_seq_len)
    test_encoded, test_dropped = filter_recipes_by_length(test_encoded, max_seq_len)

    print(f"Encoded train recipes: {len(train_encoded)} (dropped {train_dropped} over {max_seq_len} tokens)")
    print(f"Encoded val recipes: {len(val_encoded)} (dropped {val_dropped} over {max_seq_len} tokens)")
    print(f"Encoded test recipes: {len(test_encoded)} (dropped {test_dropped} over {max_seq_len} tokens)\n")

    train_lengths = [len(seq) for seq in train_encoded]
    observed_max_seq_len = max(train_lengths)
    print(
        f"Train sequence lengths - Min: {min(train_lengths)}, Max: {observed_max_seq_len}, "
        f"Avg: {sum(train_lengths) // len(train_lengths)}\n"
    )

    print("Step 6: Creating language-model batches...")
    batch_size = TRAINING_CONFIG.batch_size
    pad_id = token_to_id["<PAD>"]

    train_batches = create_recipe_lm_batches(
        train_encoded,
        batch_size,
        pad_token_id=pad_id,
        sort_by_length=True,
        shuffle=True,
    )
    val_batches = create_recipe_lm_batches(
        val_encoded,
        batch_size,
        pad_token_id=pad_id,
        sort_by_length=True,
        shuffle=False,
    )
    test_batches = create_recipe_lm_batches(
        test_encoded,
        batch_size,
        pad_token_id=pad_id,
        sort_by_length=True,
        shuffle=False,
    )

    print(f"Train batches: {len(train_batches)}")
    print(f"Val batches: {len(val_batches)}")
    print(f"Test batches: {len(test_batches)}")

    first_input_ids, first_target_ids = train_batches[0]
    print(f"First train input_ids shape: {tuple(first_input_ids.shape)}")
    print(f"First train target_ids shape: {tuple(first_target_ids.shape)}\n")

    print("Step 7: Creating model...")
    model = RecipeTransformer(
        vocab_size=len(token_to_id),
        max_seq_len=max_seq_len,
        d_model=TRAINING_CONFIG.d_model,
        num_layers=TRAINING_CONFIG.num_layers,
        num_heads=TRAINING_CONFIG.num_heads,
        d_ff=TRAINING_CONFIG.d_ff,
        dropout=TRAINING_CONFIG.dropout,
        pad_token_id=pad_id,
        eos_token_id=token_to_id["<EOS>"],
        bos_token_id=token_to_id["<BOS>"],
    )
    print(f"Model created with {model.get_num_params():,} parameters\n")

    print("Step 8: Training model...")
    total_tokens = sum(len(recipe) for recipe in train_encoded)
    eos_id = token_to_id["<EOS>"]
    eos_tokens = sum(recipe.count(eos_id) for recipe in train_encoded)
    eos_freq = eos_tokens / total_tokens
    eos_weight = TRAINING_CONFIG.eos_weight

    print(f"EOS Analysis: {eos_tokens}/{total_tokens:,} tokens ({eos_freq:.4f})")
    print(f"Using EOS weight: {eos_weight:.1f}x\n")

    train_result = simple_train(
        model=model,
        train_batches=train_batches,
        val_batches=val_batches,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        epochs=TRAINING_CONFIG.epochs,
        learning_rate=TRAINING_CONFIG.learning_rate,
        device="auto",
        eos_weight=eos_weight,
        output_dir=TRAINING_CONFIG.artifact_dir,
        sample_every_n_epochs=TRAINING_CONFIG.sample_every_n_epochs,
        checkpoint_every_n_epochs=TRAINING_CONFIG.checkpoint_every_n_epochs,
    )

    print("\nStep 9: Evaluating on held-out test split...")
    evaluator = RecipeEvaluator(model, token_to_id, id_to_token, device=next(model.parameters()).device)
    test_results = evaluator.evaluate_model(test_batches)
    evaluator.print_evaluation_report(test_results)
    print(f"Best model path: {train_result['best_model_path']}")
    print(f"Final model path: {train_result['final_model_path']}")

    print("\nTraining pipeline complete!")


if __name__ == "__main__":
    main()
