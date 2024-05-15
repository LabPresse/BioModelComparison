
# Import libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO: Add in RNN structure

# Define Mamba block
class MambaBlock(nn.Module):
    def __init__(self, n_features, n_states):
        super(MambaBlock, self).__init__()

        # Set attributes
        self.n_features = n_features
        self.n_states = n_states

        # Set up layers
        self.linear1 = nn.Linear(n_features, n_features)
        self.state_space = nn.Parameter(torch.randn(n_features, n_states))
        self.linear2 = nn.Linear(n_states, n_features)

    def forward(self, x):

        # Save original input for skip connection
        x0 = x

        # Apply a linear transformation
        x = self.linear1(x)

        # Apply nonlinearity
        x = F.relu(x)

        # Selective state space interaction
        x = torch.einsum('bij,jk->bik', x, self.state_space)

        # Another linear transformation to restore dimensionality
        x = self.linear2(x)

        # Add skip connection
        x += x0

        # Return x
        return x


# Define Mamba class
class VisionMamba(nn.Module):
    def __init__(self, 
            img_size, in_channels, out_channels,
            n_layers=8, n_states=64, n_features=64,
            patch_size=8, **kwargs
        ):
        super(VisionMamba, self).__init__()

        # Set attributes
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_states = n_states
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
        self.pos_embed = pos_embed

        # Patch embedding
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, n_features, kernel_size=patch_size, stride=patch_size),
        )

        # Set up mamba blocks
        self.mamba_blocks = nn.ModuleList()
        for i in range(n_layers):
            self.mamba_blocks.append(MambaBlock(n_features, n_states, **kwargs))

        # Patch expansion
        self.patch_expansion = nn.Sequential(
            nn.ConvTranspose2d(
                self.n_features, 
                out_channels, 
                kernel_size=patch_size, 
                stride=patch_size,
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
        for block in self.mamba_blocks:
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
    model = VisionMamba(128, 3, 1, n_layers=16, n_features=64)

    # Print number of parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Set up input tensor
    x = torch.randn(1, 3, 128, 128)

    # Test model
    y = model(x)

    # Print output shape
    print(y.shape)
        

