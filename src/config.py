from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _legacy_root_file(name: str) -> Path:
    return PROJECT_ROOT / name


def ensure_parent_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_input_path(primary: Path, legacy_name: str) -> Path:
    legacy_path = _legacy_root_file(legacy_name)
    if primary.exists():
        return primary
    if legacy_path.exists():
        return legacy_path
    return primary


@dataclass(frozen=True)
class RecipeDataConfig:
    raw_recipes_csv: Path = PROJECT_ROOT / "data" / "raw" / "RAW_recipes.csv"
    processed_recipes_txt: Path = PROJECT_ROOT / "data" / "processed" / "recipes.txt"
    tokenizer_vocab_json: Path = PROJECT_ROOT / "artifacts" / "tokenizer" / "recipe_tokenizer_vocab.json"

    def resolved_raw_recipes_csv(self) -> Path:
        return resolve_input_path(self.raw_recipes_csv, "RAW_recipes.csv")


@dataclass(frozen=True)
class TrainingConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    max_examples: int = 10000

    batch_size: int = 8
    max_seq_len: int = 640
    learning_rate: float = 2.0e-4
    epochs: int = 20
    eos_weight: float = 1.0

    d_model: int = 384
    num_layers: int = 6
    num_heads: int = 6
    d_ff: int = 1536
    dropout: float = 0.1

    sample_every_n_epochs: int = 4
    checkpoint_every_n_epochs: int = 5
    artifact_dir: Path = PROJECT_ROOT / "artifacts" / "models"


RECIPE_DATA_CONFIG = RecipeDataConfig()
TRAINING_CONFIG = TrainingConfig()
