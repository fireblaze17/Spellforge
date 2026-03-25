from src.config import RECIPE_DATA_CONFIG, TRAINING_CONFIG
from src.preprocessing import preprocess_and_save


def main():
    source_csv = RECIPE_DATA_CONFIG.resolved_raw_recipes_csv()
    output_path = RECIPE_DATA_CONFIG.processed_recipes_txt

    print(f"Preprocessing recipes from {source_csv}...")
    num_recipes = preprocess_and_save(
        csv_path=source_csv,
        output_path=output_path,
        max_recipes=TRAINING_CONFIG.max_examples,
        random_seed=TRAINING_CONFIG.split_seed,
    )
    print(f"Saved {num_recipes} recipes to {output_path}")


if __name__ == "__main__":
    main()
