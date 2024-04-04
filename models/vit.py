
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define transformer block
class TransformerBlock(nn.Module):
    def __init__(self, n_features, n_heads, mlp_ratio=2, dropout=0.1):
        super(TransformerBlock, self).__init__()

        # Set up attributes
        self.n_features = n_features
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout

        # Set up multi-head self-attention
        self.self_attn = nn.MultiheadAttention(n_features, n_heads, dropout=dropout, batch_first=True)

        # Set up feedforward layer
        self.mlp = nn.Sequential(
            nn.Linear(n_features, int(n_features * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(n_features * mlp_ratio), n_features),
            nn.Dropout(dropout),
        )

        # Set up normalization layers
        self.norm1 = nn.LayerNorm(n_features)
        self.norm2 = nn.LayerNorm(n_features)

    def forward(self, x):

        # Apply self-attention
        attn_output, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_output

        # Feedforward layer
        x = x + self.mlp(self.norm2(x))

        return x


# Define vision transformer class
class VisionTransformer(nn.Module):
    def __init__(self, 
            img_size, in_channels, out_channels,
            n_layers=4, n_features=64,
            n_heads=8, patch_size=16, use_cls_token=True, **kwargs
        ):
        super(VisionTransformer, self).__init__()

        # Set attributes
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.n_features = n_features
        self.n_heads = n_heads
        self.patch_size = patch_size
        self.use_cls_token = use_cls_token

        # Calculate constants
        n_patches = (img_size // patch_size) ** 2
        shape_after_patch = (img_size // patch_size, img_size // patch_size)
        self.n_patches = n_patches
        self.shape_after_patch = shape_after_patch

        # Set up embedding parameters
        if use_cls_token:
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, n_features))
            self.cls_token = nn.Parameter(torch.zeros(1, 1, n_features))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, n_features))

        ### Set up blocks ###

        # Patch embedding
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, n_features, kernel_size=patch_size, stride=patch_size),
        )

        # Set up transformer blocks
        self.transformer_blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.transformer_blocks.append(TransformerBlock(n_features, n_heads, **kwargs))

        # Patch expansion
        self.patch_expansion = nn.Sequential(
            nn.ConvTranspose2d(n_features, out_channels, kernel_size=patch_size, stride=patch_size),
        )

    def forward(self, x, mask=None):
        """Forward pass."""

        # Convert image to patch embeddings
        x = self.patch_embedding(x)  # Expected shape: (B, C, H, W)
        x = x.flatten(2)             # Flatten patch embeddings
        x = x.transpose(1, 2)        # Transpose for sequence-first format

        # Apply mask if provided
        if mask is not None:
            x *= mask.unsqueeze(-1)  # Mask shape should be (B, N_patches)

        # Add class token
        if self.use_cls_token:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Remove class token
        if self.use_cls_token:
            x = x[:, 1:, :]
        
        # Convert patch embeddings to image
        x = x.transpose(1, 2)    # Transpose back to sequence-last format
        x = x.view(x.shape[0], self.n_features, *self.shape_after_patch)
        x = self.patch_expansion(x)

        # Return
        return x


# Test the class
if __name__ == '__main__':
    model = VisionTransformer(128, 3, 3)
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    print(y.shape)  # Expected: (1, 3, 128, 128)
        
