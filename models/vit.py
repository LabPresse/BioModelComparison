
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the VisionTransformer class
class VisionTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers, patch_size, hidden_dim, num_heads, dropout):
        super(VisionTransformer, self).__init__()
        
        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim, dropout=dropout),
            num_layers=n_layers
        )
        
        self.segmentation_head = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer_encoder(x)
        x = self.segmentation_head(x)
        return x
