import math
from typing import Dict, List

import torch

from src.word_tokenizer import decode_tokens_words, encode_text_words


class RecipeEvaluator:
    """Evaluation utilities for the recipe generator."""

    def __init__(self, model, token_to_id, id_to_token, device="cpu"):
        self.model = model
        self.token_to_id = token_to_id
        self.id_to_token = id_to_token
        self.device = device

        self.pad_id = token_to_id.get("<PAD>", 0)
        self.bos_id = token_to_id.get("<BOS>", 1)
        self.eos_id = token_to_id.get("<EOS>", 2)

    def compute_perplexity(self, batches) -> float:
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for input_ids, target_ids in batches:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                loss, _ = self.model.forward_lm_batch(input_ids, target_ids)
                non_pad_tokens = (target_ids != self.pad_id).sum().item()
                total_loss += loss.item() * non_pad_tokens
                total_tokens += non_pad_tokens

        if total_tokens == 0:
            return float("inf")

        avg_loss = total_loss / total_tokens
        return math.exp(avg_loss)

    def compute_token_accuracy(self, batches) -> float:
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for input_ids, target_ids in batches:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                _, logits = self.model.forward_lm_batch(input_ids, target_ids)
                predictions = torch.argmax(logits, dim=-1)
                mask = target_ids != self.pad_id
                correct_predictions += ((predictions == target_ids) & mask).sum().item()
                total_predictions += mask.sum().item()

        if total_predictions == 0:
            return 0.0

        return correct_predictions / total_predictions

    def encode_prompt(self, text_prompt: str) -> torch.Tensor:
        prompt_tokens = encode_text_words(
            text_prompt,
            self.token_to_id,
            add_bos=True,
            add_eos=False,
        )
        return torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

    def decode_tokens(self, token_ids: torch.Tensor, skip_special=True) -> str:
        if token_ids.ndim > 1:
            token_ids = token_ids.squeeze()

        token_list = token_ids.tolist()
        if skip_special:
            return decode_tokens_words(token_list, self.id_to_token)

        return " ".join(self.id_to_token.get(token_id, f"<UNK:{token_id}>") for token_id in token_list)

    def generate_sample_recipes(self, num_samples=3, temperature=0.8) -> List[str]:
        sample_prompts = [
            "<<< New Recipe Forged >>>\n  Name:",
            "<<< New Recipe Forged >>>\n  Name: skillet",
            "<<< New Recipe Forged >>>\n  Name: roasted garlic pasta\n  Ingredients:\n  -",
        ]

        generated_recipes = []
        self.model.eval()

        for prompt in sample_prompts[:num_samples]:
            prompt_ids = self.encode_prompt(prompt)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    prompt_ids,
                    max_new_tokens=350,
                    temperature=temperature,
                    top_k=40,
                    top_p=0.9,
                    repetition_penalty=1.15,
                    do_sample=True,
                    eos_token_id=self.eos_id,
                )

            generated_recipes.append(self.decode_tokens(generated_ids))

        return generated_recipes

    def check_format_compliance(self, generated_text: str) -> Dict[str, bool]:
        ingredient_lines = sum(1 for line in generated_text.splitlines() if line.strip().startswith("-"))
        return {
            "starts_with_header": generated_text.strip().startswith("<<< New Recipe Forged >>>"),
            "ends_with_footer": "<<< May it Feed You Well >>>" in generated_text,
            "has_name_field": "Name:" in generated_text,
            "has_ingredients_field": "Ingredients:" in generated_text,
            "has_steps_field": "Steps:" in generated_text,
            "has_multiple_list_items": ingredient_lines >= 2,
        }

    def evaluate_model(self, batches, num_samples=3, temperature=0.8) -> Dict:
        print("Starting recipe model evaluation...")
        perplexity = self.compute_perplexity(batches)
        accuracy = self.compute_token_accuracy(batches)
        sample_recipes = self.generate_sample_recipes(num_samples=num_samples, temperature=temperature)
        format_scores = [self.check_format_compliance(recipe) for recipe in sample_recipes]

        total_checks = len(format_scores) * len(format_scores[0]) if format_scores else 0
        passed_checks = sum(sum(checks.values()) for checks in format_scores)

        return {
            "perplexity": perplexity,
            "token_accuracy": accuracy,
            "format_compliance_rate": passed_checks / total_checks if total_checks else 0.0,
            "sample_recipes": sample_recipes,
            "format_checks": format_scores,
            "num_parameters": self.model.get_num_params(),
        }

    def evaluate_generation_only(self, num_samples=3, temperature=0.8) -> Dict:
        sample_recipes = self.generate_sample_recipes(num_samples=num_samples, temperature=temperature)
        format_scores = [self.check_format_compliance(recipe) for recipe in sample_recipes]
        total_checks = len(format_scores) * len(format_scores[0]) if format_scores else 0
        passed_checks = sum(sum(checks.values()) for checks in format_scores)

        return {
            "format_compliance_rate": passed_checks / total_checks if total_checks else 0.0,
            "sample_recipes": sample_recipes,
            "format_checks": format_scores,
            "num_parameters": self.model.get_num_params(),
        }

    def print_evaluation_report(self, results: Dict):
        print("\n" + "=" * 60)
        print("RECIPE GENERATOR EVALUATION REPORT")
        print("=" * 60)
        print(f"Perplexity: {results['perplexity']:.2f}")
        print(f"Token Accuracy: {results['token_accuracy']:.1%}")
        print(f"Format Compliance: {results['format_compliance_rate']:.1%}")
        print(f"Model Parameters: {results['num_parameters']:,}")

        for i, recipe in enumerate(results["sample_recipes"], 1):
            print(f"\n--- Sample {i} ---")
            print(recipe[:300] + "..." if len(recipe) > 300 else recipe)

    def print_generation_report(self, results: Dict):
        print("\n" + "=" * 60)
        print("RECIPE GENERATION REPORT")
        print("=" * 60)
        print(f"Format Compliance: {results['format_compliance_rate']:.1%}")
        print(f"Model Parameters: {results['num_parameters']:,}")

        for i, recipe in enumerate(results["sample_recipes"], 1):
            print(f"\n--- Sample {i} ---")
            print(recipe[:300] + "..." if len(recipe) > 300 else recipe)
