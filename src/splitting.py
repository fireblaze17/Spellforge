import random


def split_dataset(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split a list into train/val/test parts.
    Returns (train_data, val_data, test_data).
    """
    if not isinstance(data, list):
        raise TypeError("data must be a list")
    if len(data) == 0:
        raise ValueError("data cannot be empty")

    for ratio_name, ratio_value in {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }.items():
        if not isinstance(ratio_value, (int, float)):
            raise TypeError(f"{ratio_name} must be a number")
        if ratio_value < 0:
            raise ValueError(f"{ratio_name} must be >= 0")

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    shuffled = data.copy()
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_data = shuffled[:train_end]
    val_data = shuffled[train_end:val_end]
    test_data = shuffled[val_end:]

    if len(train_data) == 0:
        raise ValueError("train split is empty; increase dataset size or train_ratio")
    if val_ratio > 0 and len(val_data) == 0:
        raise ValueError("val split is empty; increase dataset size or val_ratio")
    if test_ratio > 0 and len(test_data) == 0:
        raise ValueError("test split is empty; increase dataset size or test_ratio")

    return train_data, val_data, test_data
