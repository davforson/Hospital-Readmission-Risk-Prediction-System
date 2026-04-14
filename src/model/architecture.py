import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class ReadmissionPredictor(nn.Module):
    """Predictor for the 30 day readmission

    Architecture:
      Input (n features)
        → Linear(n, 128) → BatchNorm → LeakyReLU → Dropout
        → Linear(128, 64) → BatchNorm → LeakyReLU → Dropout
        → Linear(64, 32)  → BatchNorm → LeakyReLU → Dropout
        → Linear(32, 1)   → Sigmoid
        → Output (probability of readmission)
    """
    def __init__(self, input_dim: int, hidden_dims: list =[128, 64, 32], dropout_rate: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims: 
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim


        layers.append(nn.Linear(prev_dim, 1))
        # layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)