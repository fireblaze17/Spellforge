import math

import torch


# Single-head self-attention first (easier to debug before multi-head)
class SelfAttention(torch.nn.Module):
    """
    Input shape: [batch_size, seq_len, d_model]
    Output shape: [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model):
        super().__init__()

        if not isinstance(d_model, int):
            raise TypeError("d_model must be an integer")
        if d_model <= 0:
            raise ValueError("d_model must be > 0")

        self.d_model = d_model

        # linear projections for query, key, value
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask=None):
        if not torch.is_tensor(x):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError("x must have shape [batch_size, seq_len, d_model]")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"last dim of x must be d_model={self.d_model}")

        # project x to q, k, v
        q = self.q_proj(x)  # [B, T, D]
        k = self.k_proj(x)  # [B, T, D]
        v = self.v_proj(x)  # [B, T, D]

        # attention score matrix: [B, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)

        if attn_mask is not None:
            if not torch.is_tensor(attn_mask) or attn_mask.dtype != torch.bool:
                raise TypeError("attn_mask must be a bool torch.Tensor")
            if attn_mask.ndim != 4:
                raise ValueError("attn_mask must have shape [batch_size, 1, seq_len, seq_len]")

            # remove head dimension because this is single-head attention
            mask = attn_mask.squeeze(1)  # [B, T, T]
            if mask.shape != scores.shape:
                raise ValueError(
                    f"attn_mask shape after squeeze must match scores shape {tuple(scores.shape)}"
                )

            # block disallowed positions before softmax
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # [B, T, T]
        out = torch.matmul(weights, v)  # [B, T, D]
        return out


# Multi-head self-attention built from scratch (without nn.MultiheadAttention)
class MultiHeadSelfAttention(torch.nn.Module):
    """
    Input shape: [batch_size, seq_len, d_model]
    Output shape: [batch_size, seq_len, d_model]
    """

    def __init__(self, d_model, num_heads):
        super().__init__()

        if not isinstance(d_model, int):
            raise TypeError("d_model must be an integer")
        if not isinstance(num_heads, int):
            raise TypeError("num_heads must be an integer")
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # project to full D, then split into heads
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)

        # final projection after concatenating heads
        self.out_proj = torch.nn.Linear(d_model, d_model)

    def _split_heads(self, x):
        # x: [B, T, D] -> [B, H, T, Hd]
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        # x: [B, H, T, Hd] -> [B, T, D]
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, x, attn_mask=None):
        if not torch.is_tensor(x):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError("x must have shape [batch_size, seq_len, d_model]")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"last dim of x must be d_model={self.d_model}")

        # [B, T, D]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, H, T, Hd]
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # attention scores: [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            if not torch.is_tensor(attn_mask) or attn_mask.dtype != torch.bool:
                raise TypeError("attn_mask must be a bool torch.Tensor")
            if attn_mask.ndim != 4:
                raise ValueError("attn_mask must have shape [batch_size, 1, seq_len, seq_len]")
            if attn_mask.shape[0] != x.shape[0]:
                raise ValueError("attn_mask batch size must match x")
            if attn_mask.shape[-2] != x.shape[1] or attn_mask.shape[-1] != x.shape[1]:
                raise ValueError("attn_mask seq_len dimensions must match x")

            # attn_mask [B,1,T,T] is broadcast across heads to [B,H,T,T]
            scores = scores.masked_fill(~attn_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)  # [B, H, T, T]
        out = torch.matmul(weights, v)  # [B, H, T, Hd]

        out = self._merge_heads(out)  # [B, T, D]
        out = self.out_proj(out)  # [B, T, D]
        return out
