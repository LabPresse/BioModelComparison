
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the ResNetBlock class
class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features, mlp_ratio=.5, stride=1, upsample=False):
        super(ResNetBlock, self).__init__()
        
        # Set up attributes
        self.in_features = in_features
        self.out_features = out_features
        
        # Calculate constants
        mid_features = int(in_features * mlp_ratio)

        # Define convolution
        if upsample:
            convolution = nn.ConvTranspose2d(
                mid_features, mid_features, 
                kernel_size=3, stride=stride, padding=1, output_padding=1,
            )
        else:
            convolution = nn.Conv2d(
                mid_features, mid_features, 
                kernel_size=3, stride=stride, padding=1,
            )

        # Set up multi-layer perceptron
        self.mlp = nn.Sequential(
            # Bottleneck
            nn.Conv2d(in_features, mid_features, kernel_size=1, bias=False),
            # Convolution
            convolution,
            nn.InstanceNorm2d(mid_features),
            nn.ReLU(),
            # Expansion
            nn.Conv2d(mid_features, out_features, kernel_size=1, bias=False),
        )

        # Define shortcut connection
        if (in_features == out_features) and stride == 1:
            self.shortcut = nn.Identity()
        else:
            if upsample:
                self.shortcut = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_features, out_features, 
                        kernel_size=stride, stride=stride,
                    ),
                    nn.GroupNorm(1, out_features, affine=False),
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_features, out_features, 
                        kernel_size=stride, stride=stride,
                    ),
                    nn.GroupNorm(1, out_features, affine=False),
                )

    def forward(self, x):

        # Get shortcut connection
        identity = self.shortcut(x)

        # Apply convolution
        x = self.mlp(x)

        # Add shortcut connection
        x = x + identity

        # Apply ReLU
        x = F.relu(x)
        
        # Return output
        return x


# Define the ResNet class
class ResNet(nn.Module):
    def __init__(self,
            in_channels, out_channels,
            n_blocks=3, n_features=8, n_layers_per_block=2, expansion=2,
        ):
        super(ResNet, self).__init__()

        # Set attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_blocks
        self.n_features = n_features
        self.n_layers_per_block = n_layers_per_block
        self.expansion = expansion

        # Set up input block
        self.input_block = nn.Sequential(
            nn.GroupNorm(1, in_channels, affine=False),  # Normalize input
            nn.Conv2d(in_channels, n_features, kernel_size=1),
        )

        # Initialize resnet blocks
        self.blocks = nn.ModuleList()

        # Set up encoder blocks
        self.encoder_blocks = nn.ModuleList()
        n_in = n_features
        for i in range(n_blocks):
            n_out = n_in * expansion

            # Initialize layers
            layers = []

            # Downsample block
            layers.append(
                ResNetBlock(n_in, n_out, stride=2, upsample=False)
            )

            # Mixing layers
            for j in range(n_layers_per_block-1):
                layers.append(
                    ResNetBlock(n_out, n_out)
                )

            # Add to list
            self.encoder_blocks.append(nn.Sequential(*layers))

            # Update n_in
            n_in = n_out

        
        # Set up decoder blocks
        n_in = n_features * expansion ** n_blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(n_blocks):
            n_out = n_in // expansion

            # Initialize layers
            layers = []

            # Upsample block
            layers.append(
                ResNetBlock(n_in, n_out, stride=2, upsample=True)
            )

            # Mixing layers
            for j in range(n_layers_per_block-1):
                layers.append(
                    ResNetBlock(n_out, n_out)
                )

            # Add to list
            self.decoder_blocks.append(nn.Sequential(*layers))

            # Update n_in
            n_in = n_out

        # Set output layer
        self.set_output_block(out_channels)

    def set_output_block(self, out_channels):
        self.output_layer = nn.Sequential(
            nn.Conv2d(self.n_features, out_channels, kernel_size=1),
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
    model = ResNet(3, 2, n_blocks=3, n_features=8, n_layers_per_block=2)

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
        
                

