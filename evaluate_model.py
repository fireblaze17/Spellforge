import torch

from src.config import RECIPE_DATA_CONFIG, TRAINING_CONFIG, resolve_input_path
from src.evaluation import RecipeEvaluator
from src.model import RecipeTransformer
from src.word_tokenizer import load_recipe_vocabulary


def _default_checkpoint_path():
    primary = TRAINING_CONFIG.artifact_dir / f"final_model_epoch_{TRAINING_CONFIG.epochs}.pt"
    return resolve_input_path(primary, primary.name)


def main(checkpoint_path=None):
    vocab_path = resolve_input_path(
        RECIPE_DATA_CONFIG.tokenizer_vocab_json,
        "tokenizer_vocab.json",
    )
    token_to_id, id_to_token = load_recipe_vocabulary(vocab_path)

    model = RecipeTransformer(
        vocab_size=len(token_to_id),
        max_seq_len=TRAINING_CONFIG.max_seq_len,
        d_model=TRAINING_CONFIG.d_model,
        num_layers=TRAINING_CONFIG.num_layers,
        num_heads=TRAINING_CONFIG.num_heads,
        d_ff=TRAINING_CONFIG.d_ff,
        dropout=TRAINING_CONFIG.dropout,
        pad_token_id=token_to_id.get("<PAD>", 0),
        eos_token_id=token_to_id.get("<EOS>", 2),
        bos_token_id=token_to_id.get("<BOS>", 1),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    checkpoint_path = checkpoint_path or _default_checkpoint_path()
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except FileNotFoundError:
        fallback = resolve_input_path(TRAINING_CONFIG.artifact_dir / "best_model.pt", "best_model.pt")
        model.load_state_dict(torch.load(fallback, map_location=device))
        checkpoint_path = fallback

    print(f"Loaded model: {checkpoint_path}")
    evaluator = RecipeEvaluator(model, token_to_id, id_to_token, device=device)
    results = evaluator.evaluate_generation_only()
    evaluator.print_generation_report(results)


if __name__ == "__main__":
    main()
