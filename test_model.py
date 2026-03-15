"""
Test the trained model - comprehensive but focused evaluation.
Updated for word-level tokenization.
"""

import torch

from src.word_tokenizer import decode_tokens_words, load_word_vocabulary
from src.model import SpellforgeTransformer


def test_model(checkpoint_path="final_model_epoch_30.pt"):
    """Test the trained model thoroughly."""
    print("Testing trained Spellforge model...")

    token_to_id, id_to_token = load_word_vocabulary("tokenizer_vocab.json")

    model = SpellforgeTransformer(
        vocab_size=len(token_to_id),
        max_seq_len=1024,
        d_model=384,
        num_layers=6,
        num_heads=6,
        d_ff=1536,
        dropout=0.1,
        pad_token_id=token_to_id.get("<PAD>", 0),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model: {checkpoint_path}")
    except FileNotFoundError:
        print(f"{checkpoint_path} not found, trying best_model.pt...")
        try:
            model.load_state_dict(torch.load("best_model.pt", map_location=device))
            print("Loaded model: best_model.pt")
        except FileNotFoundError:
            print("No trained model found. Run main.py first.")
            return

    model.eval()
    print(f"Model parameters: {model.get_num_params():,}")

    print("\nGENERATION QUALITY TEST")
    print("=" * 50)

    test_configs = [
        {"temp": 0.7, "top_p": 0.9, "name": "Balanced"},
        {"temp": 0.6, "top_p": 0.95, "name": "Conservative"},
        {"temp": 0.8, "top_p": 0.85, "name": "Creative"},
    ]

    all_results = []

    for config in test_configs:
        print(f"\n{config['name']} Settings (T={config['temp']}, P={config['top_p']})")
        print("-" * 40)

        config_results = []
        for i in range(3):
            with torch.no_grad():
                prompt_ids = torch.tensor([[token_to_id.get("<BOS>", 1)]], device=device)
                generated = model.generate(
                    prompt_ids=prompt_ids,
                    max_new_tokens=300,
                    temperature=config["temp"],
                    top_p=config["top_p"],
                    do_sample=True,
                    eos_token_id=token_to_id.get("<EOS>", 2),
                )

            tokens = generated.squeeze().tolist()
            text = decode_tokens_words(tokens, id_to_token)

            analysis = analyze_spell_quality(text, tokens, token_to_id)
            config_results.append(analysis)
            all_results.append({**analysis, "config": config["name"]})

            print(f"Sample {i + 1}:")
            display_text = text[:350] + "..." if len(text) > 350 else text
            print(display_text)
            print(
                f"Complete: {analysis['complete']} | Structure: {analysis['structured']} | "
                f"Words: {analysis['word_count']} | EOS: {analysis['has_eos_token']}"
            )
            print()

        complete_rate = sum(r["complete"] for r in config_results) / len(config_results) * 100
        print(f"{config['name']} Summary: {complete_rate:.0f}% completion rate")

    print("\nOVERALL PERFORMANCE")
    print("=" * 50)

    total_samples = len(all_results)
    completion_rate = sum(r["complete"] for r in all_results) / total_samples * 100
    structure_rate = sum(r["structured"] for r in all_results) / total_samples * 100
    eos_rate = sum(r["has_eos_token"] for r in all_results) / total_samples * 100
    avg_words = sum(r["word_count"] for r in all_results) / total_samples

    print(f"Completion Rate: {completion_rate:.1f}%")
    print(f"Structure Rate: {structure_rate:.1f}%")
    print(f"EOS Generation: {eos_rate:.1f}%")
    print(f"Average Words: {avg_words:.1f}")

    complete_spells = [r for r in all_results if r["complete"] and r["structured"]]
    if complete_spells:
        print(f"\nBEST COMPLETE SPELLS ({len(complete_spells)} found)")
        print("=" * 50)
        for i, spell in enumerate(complete_spells[:2]):
            print(f"\nComplete Spell #{i + 1} ({spell['config']} settings):")
            print("-" * 40)
            print(spell["text"])
    else:
        print("\nNo fully complete spells found in this test")

    print("\nTest complete.")


def analyze_spell_quality(text, tokens, token_to_id):
    """Analyze the quality of a generated spell."""
    has_eos_token = token_to_id.get("<EOS>", 2) in tokens
    is_complete = "<<< May it Serve You Well >>>" in text
    word_count = len(text.split())

    required_fields = ["Name:", "Classes:", "School:", "Range:", "Duration:", "Description:"]
    has_structure = all(field in text for field in required_fields)

    has_header = "<<< New Spell Forged >>>" in text
    has_footer = "<<< May it Serve You Well >>>" in text

    return {
        "text": text,
        "complete": is_complete,
        "structured": has_structure,
        "has_eos_token": has_eos_token,
        "has_header": has_header,
        "has_footer": has_footer,
        "word_count": word_count,
    }


if __name__ == "__main__":
    test_model()
