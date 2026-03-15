import torch


# Feed Forward Network used inside each transformer block
# runs on each token independently
# input shape: [batch_size, seq_len, d_model]
# output shape: [batch_size, seq_len, d_model]
class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff=None, activation="gelu", dropout=0.0, bias=True):
        super().__init__()

        if not isinstance(d_model, int):
            raise TypeError("d_model must be an integer")
        if d_model <= 0:
            raise ValueError("d_model must be > 0")

        # common default in transformers
        if d_ff is None:
            d_ff = 4 * d_model
        if not isinstance(d_ff, int):
            raise TypeError("d_ff must be an integer")
        if d_ff <= 0:
            raise ValueError("d_ff must be > 0")

        if not isinstance(activation, str):
            raise TypeError("activation must be a string")
        activation = activation.lower()
        if activation not in ("relu", "gelu"):
            raise ValueError("activation must be 'relu' or 'gelu'")

        if not isinstance(dropout, (int, float)):
            raise TypeError("dropout must be a number")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("dropout must be in [0.0, 1.0]")

        if not isinstance(bias, bool):
            raise TypeError("bias must be a bool")

        self.d_model = d_model
        self.d_ff = d_ff
        self.activation_name = activation

        # D -> d_ff -> D
        self.fc1 = torch.nn.Linear(d_model, d_ff, bias=bias)
        self.fc2 = torch.nn.Linear(d_ff, d_model, bias=bias)

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        else:
            self.activation = torch.nn.GELU()

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        if not torch.is_tensor(x):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 3:
            raise ValueError("x must have shape [batch_size, seq_len, d_model]")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"last dim of x must be d_model={self.d_model}")

        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
