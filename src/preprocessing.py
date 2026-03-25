# Starting of preprocessing phase
# The recipe file has features: name, id, minutes, contributor_id, submitted, tags, nutrition, n_steps, steps, description, ingredients, n_ingredients
# Out of all of these, we will only keep name, ingredients, steps, and description when it is actually useful

import ast
import random
import re
from pathlib import Path

import pandas as pd


def _safe_parse_list(value):
    """
    Parse the stringified list columns safely.
    Returns [] if the value is missing or malformed.
    """
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value

    text = str(value).strip()
    if text == "":
        return []

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []

    if not isinstance(parsed, list):
        return []
    return parsed


def _clean_text(text):
    """
    Normalize whitespace and remove line breaks so the training text is consistent.
    """
    if pd.isna(text):
        return ""

    cleaned = str(text).replace("\r", " ").replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _clean_list_items(values):
    """
    Clean text inside list fields like ingredients and steps.
    Drops blank values after cleaning.
    """
    cleaned_values = []
    for value in values:
        cleaned = _clean_text(value)
        if cleaned != "":
            cleaned_values.append(cleaned)
    return cleaned_values


def _normalize_ingredient_text(text):
    """
    Normalize ingredient text before comparing duplicates.
    """
    cleaned = _clean_text(text).lower()
    cleaned = cleaned.replace(" - ", "-")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.strip(" ,.;:")


def _dedupe_ingredients(values):
    """
    Remove duplicate ingredients while keeping the original order.
    """
    deduped_values = []
    seen = set()

    for value in values:
        normalized = _normalize_ingredient_text(value)
        if normalized == "":
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped_values.append(normalized)

    return deduped_values


def _description_is_useful(text):
    """
    Decide if a description is worth keeping.
    The row does not get dropped if the description is bad, we just omit the field.
    """
    if text == "":
        return False

    lowered = text.lower()
    if lowered in {"n/a", "na", "none", "null", "no description", "nothing", "nil"}:
        return False

    if len(text) < 30:
        return False

    if len(text.split()) < 6:
        return False

    alpha_chars = sum(ch.isalpha() for ch in text)
    if alpha_chars < len(text) * 0.6:
        return False

    return True


def preprocess_and_save(csv_path, output_path, max_recipes=None, random_seed=42):
    """
    Load data as a pandas dataframe and prepare formatted recipes.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Loading data as a pandas dataframe
    data = pd.read_csv(csv_path)

    # Keeping only relevant columns
    data = data[["name", "steps", "description", "ingredients"]]

    # Need to decide on a format for each recipe
    # Format:
    # <<< New Recipe Forged >>>
    #     Name:
    #     Ingredients:
    #     Steps:
    #     Description:  (only when useful)
    #     <<< May it Feed You Well >>>

    # List to hold formatted recipes
    formatted_recipes = []

    # For loop to parse/clean rows and then append a string with the recipes to the list
    for row in data.itertuples(index=False):
        name = _clean_text(row.name)
        ingredients = _dedupe_ingredients(_clean_list_items(_safe_parse_list(row.ingredients)))
        steps = _clean_list_items(_safe_parse_list(row.steps))
        description = _clean_text(row.description)

        # Dropping rows with missing required values
        if name == "" or len(ingredients) == 0 or len(steps) == 0:
            continue

        # Extra cleanup so bad rows do not sneak through
        if len(name.split()) < 2:
            continue

        # Build each section of the recipe format
        ingredient_lines = "\n".join(f"  - {ingredient}" for ingredient in ingredients)
        step_lines = "\n".join(f"  - {step}" for step in steps)

        recipe_parts = [
            "<<< New Recipe Forged >>>",
            f"  Name: {name}",
            "  Ingredients:",
            ingredient_lines,
            "  Steps:",
            step_lines,
        ]

        # Description is optional, so keep it only when it adds value
        if _description_is_useful(description):
            recipe_parts.append(f"  Description: {description}")

        recipe_parts.append("<<< May it Feed You Well >>>")
        recipe_str = "\n".join(recipe_parts)
        formatted_recipes.append(recipe_str)

    # Join recipes with blank lines between them so the file is readable and boundaries stay clear
    if max_recipes is not None and len(formatted_recipes) > max_recipes:
        rng = random.Random(random_seed)
        formatted_recipes = rng.sample(formatted_recipes, max_recipes)

    final_recipes = "\n\n".join(formatted_recipes)

    # Writing to text file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_recipes)

    return len(formatted_recipes)
