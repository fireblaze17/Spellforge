import torch


def _pad_sequences(sequences, pad_token_id):
    """
    Pad a list of token ID sequences to the max length in that list.
    Returns tensor of shape [batch_size, max_seq_len].
    """
    if not isinstance(sequences, list):
        raise TypeError("sequences must be a list")
    if len(sequences) == 0:
        raise ValueError("sequences cannot be empty")

    max_len = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        if not isinstance(seq, list):
            raise TypeError("each sequence must be a list")
        if len(seq) == 0:
            raise ValueError("sequences cannot be empty")
        padded.append(seq + [pad_token_id] * (max_len - len(seq)))

    return torch.tensor(padded, dtype=torch.long)


def create_lm_batch(encoded_batch, pad_token_id=0):
    """
    Build one next-token LM batch from encoded sequences.

    Input:
    - encoded_batch: list of encoded sequences, each with shape [seq_len]

    Returns:
    - input_ids:  [batch_size, seq_len-1] (padded in-batch)
    - target_ids: [batch_size, seq_len-1] (padded in-batch)
    """
    if not isinstance(encoded_batch, list):
        raise TypeError("encoded_batch must be a list")
    if len(encoded_batch) == 0:
        raise ValueError("encoded_batch cannot be empty")

    input_sequences = []
    target_sequences = []

    for seq in encoded_batch:
        if not isinstance(seq, list):
            raise TypeError("each sequence must be a list")
        if len(seq) < 2:
            raise ValueError("each sequence must have at least 2 tokens for LM shift")

        # teacher forcing shift:
        # input is everything except the last token
        # target is everything except the first token
        input_sequences.append(seq[:-1])
        target_sequences.append(seq[1:])

    input_ids = _pad_sequences(input_sequences, pad_token_id=pad_token_id)
    target_ids = _pad_sequences(target_sequences, pad_token_id=pad_token_id)
    return input_ids, target_ids


def create_recipe_lm_batches(encoded_recipes, batch_size, pad_token_id=0, sort_by_length=True, shuffle=True):
    """
    Build many LM batches.
    Returns a list of tuples: (input_ids, target_ids).
    """
    if not isinstance(encoded_recipes, list):
        raise TypeError("encoded_recipes must be a list")
    if len(encoded_recipes) == 0:
        raise ValueError("encoded_recipes cannot be empty")
    if not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    recipe_data = [seq.copy() for seq in encoded_recipes]
    if sort_by_length:
        recipe_data.sort(key=len)

    batches = []
    for i in range(0, len(recipe_data), batch_size):
        chunk = recipe_data[i : i + batch_size]
        batches.append(create_lm_batch(chunk, pad_token_id=pad_token_id))

    if shuffle:
        # preserve in-batch length similarity while randomizing batch order each epoch
        perm = torch.randperm(len(batches)).tolist()
        batches = [batches[i] for i in perm]

    return batches


def create_lm_batches(encoded_recipes, batch_size, pad_token_id=0, sort_by_length=True, shuffle=True):
    """Compatibility wrapper for older call sites."""
    return create_recipe_lm_batches(
        encoded_recipes,
        batch_size,
        pad_token_id=pad_token_id,
        sort_by_length=sort_by_length,
        shuffle=shuffle,
    )
