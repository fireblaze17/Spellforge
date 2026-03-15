import math

import torch

#These concepts are harder for me right now to put into code, so it takes some time and may have logic errors
#mathematically i understand what is happening, but putting it into code is a skill that is still being developed
#torch has been used to use GPU processing and torch libraries that will make the project easier, learning rate etc but not take away from learning how a transformer model is made
def get_sinusoidal_positional_embeddings(seq_len, d_model, device=None):
    """
    Returns positional embeddings of shape [1, seq_len, d_model]
    so they can be added to token embeddings of shape [batch, seq_len, d_model].
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if d_model <= 0:
        raise ValueError("d_model must be > 0")

    # [seq_len, 1]
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)

    # [d_model/2] for even dimensions (0, 2, 4, ...)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32, device=device)
        * (-math.log(10000.0) / d_model)
    )

    # [seq_len, d_model]
    pe = torch.zeros(seq_len, d_model, dtype=torch.float32, device=device)

    # Even dimensions -> sin
    pe[:, 0::2] = torch.sin(position * div_term)

    # Odd dimensions -> cos
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

    # Add batch dimension so shape becomes [1, seq_len, d_model]
    return pe.unsqueeze(0)


class TokenEmbedding(torch.nn.Module):
    """
    Learned token embeddings using one shared table for the full vocabulary.
    Input shape: [batch_size, seq_len]
    Output shape: [batch_size, seq_len, d_model]
    """

    def __init__(self, vocab_size, d_model):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if d_model <= 0:
            raise ValueError("d_model must be > 0")

        self.vocab_size = vocab_size
        self.d_model = d_model

        # nn.Embedding is basically a learnable matrix + row lookup
        # shape should be [vocab_size, d_model]
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, token_ids):
        if not torch.is_tensor(token_ids):
            raise TypeError("token_ids must be a torch.Tensor")
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch_size, seq_len]")
        if token_ids.dtype not in (torch.int64, torch.int32, torch.long, torch.int):
            raise TypeError("token_ids must be an integer tensor")
        if token_ids.numel() == 0:
            raise ValueError("token_ids cannot be empty")
        if torch.any(token_ids < 0) or torch.any(token_ids >= self.vocab_size):
            raise ValueError(f"token_ids must be in range [0, {self.vocab_size - 1}]")

        # token_ids shape: [batch_size, seq_len]
        # output shape should be: [batch_size, seq_len, d_model]
        token_embeddings = self.embedding(token_ids)

        # these embeddings represent what token it is.
        # positional embeddings represent where the token is.
        # kept separate here so this can later be added to positional embeddings outside this class.
        return token_embeddings


