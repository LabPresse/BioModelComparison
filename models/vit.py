
# Import libraries
import math
import torch
import torch.nn as nn


# Define transformer block
class TransformerBlock(nn.Module):
    def __init__(self, n_features, n_heads=8, expansion=1):
        super(TransformerBlock, self).__init__()

        # Set up attributes
        self.n_features = n_features
        self.n_heads = n_heads
        self.expansion = expansion
        
        # Calculate constants
        n_features_inner = int(n_features * expansion)
        self.n_features_inner = n_features_inner

        # Set up multi-head self-attention
        self.self_attn = nn.MultiheadAttention(n_features, n_heads, batch_first=True)

        # Set up feedforward layer
        self.mlp = nn.Sequential(
            nn.Linear(n_features, n_features_inner),
            nn.ReLU(),
            nn.Linear(n_features_inner, n_features),
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
            n_layers=8, n_features=64,
            patch_size=8, **kwargs
        ):
        super(VisionTransformer, self).__init__()

        # Set attributes
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_features = n_features

        # Calculate constants
        n_patches = (img_size // patch_size) ** 2
        shape_after_patch = (img_size // patch_size, img_size // patch_size)
        self.n_patches = n_patches
        self.shape_after_patch = shape_after_patch


        # Set up positional encoding
        pos_embed = torch.zeros(n_patches, n_features)
        position = torch.arange(0, n_patches).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_features, 2) * -(math.log(10000.0) / n_features))
        pos_embed[:, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 1::2] = torch.cos(position * div_term)
        pos_embed = pos_embed.unsqueeze(0)
        self.register_buffer('pos_embed', pos_embed)

        # Patch embedding
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, n_features, kernel_size=patch_size, stride=patch_size),
        )

        # Set up transformer blocks
        self.transformer_blocks = nn.ModuleList()
        for i in range(n_layers):
            self.transformer_blocks.append(TransformerBlock(n_features, **kwargs))

        # Set up output block
        self.set_output_block(out_channels)

    def set_output_block(self, out_channels):
        self.patch_expansion = nn.Sequential(
            nn.ConvTranspose2d(
                self.n_features, 
                out_channels, 
                kernel_size=self.patch_size, 
                stride=self.patch_size,
            ),
        )
    
    def forward(self, x):
        """Forward pass."""

        # Convert image to patch embeddings
        x = self.patch_embedding(x)  # Expected shape: (B, C, H, W)
        x = x.flatten(2)             # Flatten patch embeddings to (B, C, N)
        x = x.transpose(1, 2)        # Transpose for sequence-first format: (B, N, C)

        # Add positional embedding
        x = x + self.pos_embed

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Convert patch embeddings to image
        x = x.transpose(1, 2)    # Transpose back to sequence-last format
        x = x.view(x.shape[0], self.n_features, *self.shape_after_patch)
        x = self.patch_expansion(x)

        # Return
        return x


# Test model
if __name__ == '__main__':
    
    # Set up model
    model = VisionTransformer(128, 3, 1, n_layers=16, n_features=64)

    # Print number of parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Set up input tensor
    x = torch.randn(1, 3, 128, 128)

    # Test model
    y = model(x)

    # Test backprop
    y.sum().backward()

    # Print output shape
    print(y.shape)
        
