
# Import libraries
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum



# Define Mamba block
class MambaBlock(nn.Module):
    def __init__(self, n_features, n_states, expansion=1):
        super(MambaBlock, self).__init__()

        # Set attributes
        self.n_features = n_features
        self.n_states = n_states
        self.expansion = expansion

        # Calculate constants
        n_features_inner = int(n_features * expansion)
        n_delta = math.ceil(n_features / 16)
        self.n_features_inner = n_features_inner
        self.n_delta = n_delta

        # Initialize projection blocks
        self.project_in = nn.Sequential(
            nn.GroupNorm(1, n_features, affine=False),  # Normalize input
            nn.Linear(n_features, n_features_inner)
        )
        self.project_out = nn.Linear(n_features_inner, n_features)

        # Initialize convolution block
        self.conv_block = nn.Sequential(
            nn.Conv1d(
                n_features_inner,
                n_features_inner,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),
        )

        # Initialize tensors
        A = torch.arange(1, n_states + 1).unsqueeze(0).repeat(n_features_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(n_features_inner))

        # Initialize get tensor blocks
        self.get_C = nn.Linear(n_features_inner, n_states)
        self.get_B = nn.Linear(n_features_inner, n_states)
        self.get_Delta = nn.Sequential(
            nn.Linear(n_features_inner, n_delta),
            nn.Linear(n_delta, n_features_inner),
            nn.Softplus(),
        )

    def forward(self, x):

        # Save original input for skip connection
        x0 = x

        ### START MAMBA ###

        # Project into inner dimension
        h = self.project_in(x)
        h0 = h  # Save for residual connection

        # Apply convolution over sequence
        h = rearrange(h, 'b l f -> b f l')
        h = self.conv_block(h)
        h = rearrange(h, 'b f l -> b l f')

        # Apply SSM
        h = self.ssm(h)

        # Apply residual connection
        h = h * F.silu(h0)

        # Project back
        x = self.project_out(h)

        # Apply residual connection
        x = x + x0

        # Return x
        return x

    
    def ssm(self, h):

        # Get tensors
        A = torch.exp(self.A_log)  # (n_features_inner, n_states)
        B = self.get_B(h)          # (B, L, n_states)
        C = self.get_C(h)          # (B, L, n_states)
        D = self.D                 # (n_features_inner)
        Delta = self.get_Delta(h)  # (B, L, n_features_inner)

        # Calculate constants
        A_bar = torch.exp(-einsum(Delta, A, 'b l f, f n -> b l f n'))
        B_bar_h = einsum(Delta, B, h, 'b l f, b l n, b l f -> b l f n')

        # Perform selective scan
        ys = []
        state = torch.zeros(h.shape[0], self.n_features_inner, self.n_states, device=h.device)
        for i in range(h.shape[1]):
            state = A_bar[:, i] * state + B_bar_h[:, i]
            y = einsum(state, C[:, i], 'b f n, b n -> b f')
            ys.append(y)
        y = torch.stack(ys, dim=1)

        # Add D term
        y = y + h * D

        # Return y
        return y


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
    model = VisionMamba(128, 3, 1, n_layers=8, n_features=64)

    # Print number of parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Set up input tensor
    x = torch.randn(32, 3, 128, 128)

    # Test model
    y = model(x)

    # Print output shape
    print(y.shape)
        

