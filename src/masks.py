import torch


def make_causal_mask(seq_len, device=None):
    # causal mask for self-attention
    # each token can attend only to itself and tokens before it
    # output shape: [1, 1, seq_len, seq_len]
    # True means allowed, False means blocked
    if not isinstance(seq_len, int):
        raise TypeError("seq_len must be an integer")
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")

    # lower triangular matrix so future positions are blocked
    base = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    return base.unsqueeze(0).unsqueeze(0)


def make_padding_mask(input_ids, pad_id):
    # padding mask for attention
    # marks real tokens as True and PAD tokens as False
    # input shape: [batch_size, seq_len]
    # output shape: [batch_size, 1, 1, seq_len]
    if not torch.is_tensor(input_ids):
        raise TypeError("input_ids must be a torch.Tensor")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch_size, seq_len]")
    if input_ids.numel() == 0:
        raise ValueError("input_ids cannot be empty")
    if input_ids.dtype not in (torch.int64, torch.int32, torch.long, torch.int):
        raise TypeError("input_ids must be an integer tensor")
    if not isinstance(pad_id, int):
        raise TypeError("pad_id must be an integer")

    mask = input_ids != pad_id
    return mask.unsqueeze(1).unsqueeze(1)


def combine_masks(causal_mask, padding_mask):
    # combines both masks so attention is allowed only when:
    # 1) token is not attending to the future
    # 2) key position is not PAD
    # output shape after broadcast: [batch_size, 1, seq_len, seq_len]
    if not torch.is_tensor(causal_mask) or causal_mask.dtype != torch.bool:
        raise TypeError("causal_mask must be a bool torch.Tensor")
    if not torch.is_tensor(padding_mask) or padding_mask.dtype != torch.bool:
        raise TypeError("padding_mask must be a bool torch.Tensor")
    if causal_mask.ndim != 4:
        raise ValueError("causal_mask must have 4 dimensions")
    if padding_mask.ndim != 4:
        raise ValueError("padding_mask must have 4 dimensions")

    return causal_mask & padding_mask
