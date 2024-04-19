
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn


# Define the ResNetBlock class
class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features, bottleneck=False):
        super(ResNetBlock, self).__init__()
        
        # Set up attributes
        self.in_features = in_features
        self.out_features = out_features
        self.bottleneck = bottleneck

        # Define convolution
        if bottleneck:
            # Use bottleneck architecture
            mid_features = out_features // 4
            self.conv = nn.Sequential(
                # 1x1 convolution for bottleneck
                nn.Conv2d(
                    in_features, mid_features, 
                    kernel_size=1, bias=False
                ),
                nn.InstanceNorm2d(mid_features),
                nn.ReLU(),
                # 3x3 convolution for spatial features
                nn.Conv2d(
                    mid_features, mid_features, 
                    kernel_size=3, padding=1, bias=False
                ),
                nn.InstanceNorm2d(mid_features),
                nn.ReLU(),
                # 1x1 convolution for expansion
                nn.Conv2d(
                    mid_features, out_features, 
                    kernel_size=1, bias=False
                ),
                nn.InstanceNorm2d(out_features),
            )
        else:
            # Use standard architecture
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_features, out_features, 
                    kernel_size=3, padding=1, bias=False
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(),
                nn.Conv2d(
                    out_features, out_features, 
                    kernel_size=3, padding=1, bias=False
                ),
                nn.InstanceNorm2d(out_features),
            )

        # Define shortcut connection
        if in_features != out_features:
            self.shortcut = nn.Conv2d(
                in_features, out_features, 
                kernel_size=1, bias=False,
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
            n_blocks=4, n_features=8, n_layers_per_block=2,
            expansion=2, bottleneck=True, **kwargs
        ):
        super(ResNet, self).__init__()

        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_blocks
        self.n_features = n_features
        self.bottleneck = bottleneck

        # Set up input block
        self.input_block = nn.Sequential(
            nn.Conv2d(in_channels, n_features, kernel_size=1),
        )

        # Initialize resnet blocks
        self.blocks = nn.ModuleList()
        
        # Encode layers
        n_in = n_features
        for i in range(n_blocks//2):
            n_out = n_in * expansion
            block = nn.Sequential(
                ResNetBlock(n_in, n_out, bottleneck=bottleneck),
                *[ResNetBlock(n_out, n_out, bottleneck=bottleneck) for _ in range(n_layers_per_block-1)]
            )
            self.blocks.append(block)
            n_in = n_out

        # Decode layers
        for i in range(n_blocks//2, n_blocks):
            n_out = n_in // expansion
            block = nn.Sequential(
                ResNetBlock(n_in, n_out, bottleneck=bottleneck),
                *[ResNetBlock(n_out, n_out, bottleneck=bottleneck) for _ in range(n_layers_per_block-1)]
            )
            self.blocks.append(block)
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



# Test the model
if __name__ == '__main__':
    
    # Set up model
    model = ResNet(3, 2, n_blocks=8, n_features=8, n_layers_per_block=2, expansion=2, bottleneck=True)
    print(model)

    # Set up input tensor
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    print(x.shape, '->', y.shape)
        
                

