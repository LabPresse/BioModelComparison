
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the ResNetBlock class
class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features, stride=1, kernel_size=3):
        super(ResNetBlock, self).__init__()
        
        # Set up attributes
        self.in_features = in_features
        self.out_features = out_features
        self.stride = stride

        # Calculate constant
        padding = kernel_size // 2

        # Define convolution
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_features, out_features, 
                kernel_size=kernel_size, stride=stride, padding=padding, bias=False
            ),
            nn.GroupNorm(1, out_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_features, out_features, 
                kernel_size=kernel_size, stride=1, padding=padding, bias=False
            ),
            nn.GroupNorm(1, out_features),
        )

        # Define shortcut connection
        if stride != 1 or in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, stride=stride, bias=False),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        # Get shortcut connection
        short = self.shortcut(x)

        # Apply convolution
        x = self.conv(x)

        # Add shortcut connection
        x = x + short

        # Apply ReLU
        x = F.relu(x)
        
        # Return output
        return x


# Define the ResNet class
class ResNet(nn.Module):
    def __init__(self,
            in_channels, out_channels,
            n_layers=8, n_blocks_per_layer=2, n_features=64,
            expansion=2, bottleneck=False, **kwargs
        ):
        super(ResNet, self).__init__()

        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.n_features = n_features
        self.bottleneck = bottleneck

        # Set up input block
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, n_features, kernel_size=1),
        )

        # Initialize resnet blocks
        self.blocks = nn.ModuleList()

        # Add encoder layers
        n_in = n_features
        for i in range(0, n_layers // 2):
            n_out = n_in * expansion
            for j in range(n_blocks_per_layer):
                self.blocks.append(ResNetBlock(n_in, n_out))
                n_in = n_out

        # Add bottleneck layer
        n_mid = n_in // 4
        if bottleneck:
            self.blocks.append(nn.Sequential(
                ResNetBlock(n_in, n_mid, kernel_size=1),
                ResNetBlock(n_mid, n_mid),
                ResNetBlock(n_mid, n_out, kernel_size=1),
            ))

        # Add decoder layers
        for i in range(n_layers // 2, n_layers):
            n_out = n_in // expansion
            for j in range(n_blocks_per_layer):
                self.blocks.append(ResNetBlock(n_in, n_out))
                n_in = n_out

        # Set output layer
        self.set_output_layer(out_channels)

    def set_output_layer(self, n_channels):
        """Set the output layer to have the given number of channels."""
        self.output_layer = nn.Sequential(
            nn.Conv2d(self.n_features, n_channels, kernel_size=1),
        )

    def forward(self, x):
            
        # Apply input block
        x = self.input_block(x)

        # Apply resnet blocks
        for block in self.blocks:
            x = block(x)

        # Apply output layer
        x = self.output_layer(x)

        # Return output
        return x




        
                

