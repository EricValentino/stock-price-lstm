import torch
import torch.nn as nn


class PredictionModel(nn.Module):
    """Stacked LSTM that maps a window of past prices to the next price.

    Input shape: ``(batch, window, input_dim)``. Output shape: ``(batch, output_dim)``.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            # PyTorch only applies dropout between stacked layers.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"Expected input of shape (batch, window, features), got {tuple(x.shape)}"
            )
        # Hidden and cell states default to zeros when not provided.
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
