import torch

from src.attention import MultiHeadSelfAttention
from src.ffn import FeedForward


# one transformer block:
# 1) pre-norm attention + residual
# 2) pre-norm ffn + residual
class TransformerBlock(torch.nn.Module):
    """
    Input shape: [batch_size, seq_len, d_model]
    Output shape: [batch_size, seq_len, d_model]
    """

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff=None,
        activation="gelu",
        dropout=0.0,
        bias=True,
    ):
        super().__init__()

        if not isinstance(d_model, int):
            raise TypeError("d_model must be an integer")
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if not isinstance(num_heads, int):
            raise TypeError("num_heads must be an integer")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")

        self.d_model = d_model

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
        self.attn_dropout = torch.nn.Dropout(dropout)

        self.norm2 = torch.nn.LayerNorm(d_model)
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout=dropout,
            bias=bias,
        )
        self.ffn_dropout = torch.nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        if not torch.is_tensor(x):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError("x must have shape [batch_size, seq_len, d_model]")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"last dim of x must be d_model={self.d_model}")

        # pre-norm attention branch
        attn_out = self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.attn_dropout(attn_out)

        # pre-norm ffn branch
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.ffn_dropout(ffn_out)

        return x
