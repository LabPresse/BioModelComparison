
# Import libraries
import torch
import torch.nn as nn


# Define the ConvolutionalNet class
class ConvolutionalNet(nn.Module):
    def __init__(self, 
            in_channels, out_channels, 
            n_features=8, n_layers=16
        ):
        super(ConvolutionalNet, self).__init__()

        # Set up attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.n_layers = n_layers


        ### SET UP BLOCKS ###

        # Set up input block
        self.input_block = nn.Sequential(
            nn.InstanceNorm2d(in_channels),  # Normalize input
            nn.Conv2d(in_channels, n_features, kernel_size=3, padding=1),
        )

        # Set up layers
        layers = []
        for i in range(n_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(n_features, n_features, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(n_features),
                    nn.ReLU(),
                )
            )
        self.layers = nn.Sequential(*layers)

        # Set up output block
        self.output_block = nn.Sequential(
            nn.Conv2d(n_features, out_channels, kernel_size=1),
        )

    def forward(self, x):
        """Forward pass."""

        # Input block
        x = self.input_block(x)

        # Layers
        x = self.layers(x)

        # Output block
        x = self.output_block(x)

        # Return
        return x


# Test model
if __name__ == '__main__':
    
    # Set up model
    model = ConvolutionalNet(3, 1, n_features=8, n_layers=16)

    # Print number of parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Set up input tensor
    x = torch.randn(1, 3, 256, 256)

    # Test model
    y = model(x)

    # Print output shape
    print(y.shape)

