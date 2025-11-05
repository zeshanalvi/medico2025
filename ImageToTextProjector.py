# Optional: project the image features to textual embedding space
import torch.nn as nn
class ImageToTextProjector(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),  # reduce to text embedding dimension
        )

    def forward(self, x):
        return self.proj(x)
