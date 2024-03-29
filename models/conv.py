
# Import libraries
import torch
import torch.nn as nn


# Define the ConvolutionalNet class
class ConvolutionalNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_features=8, n_layers=16, activation=nn.GELU()):
        super(ConvolutionalNet, self).__init__()

        # Set up attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.n_layers = n_layers


        ### SET UP BLOCKS ###

        # Set up input block
        self.input_block = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, n_features, kernel_size=3, padding=1),
        )

        # Set up layers
        self.layers = nn.ModuleList()
        for i in range(n_layers):

            # Initialize layers
            layers = []

            # Convolutional layer
            n_in = n_features
            n_out = n_features
            layers.append(
                nn.Sequential(
                    nn.Conv2d(n_in, n_out, kernel_size=3, padding=1),
                    nn.GroupNorm(1, n_out),
                    activation,
                )
            )

            # Add to list
            self.layers.append(nn.Sequential(*layers))

        # Set up output block
        self.output_block = nn.Sequential(
            nn.Conv2d(n_features, out_channels, kernel_size=1),
        )

    def forward(self, x):
            
        # Initialize
        skips = []

        # Input block
        x = self.input_block(x)

        # Encoder blocks
        for block in self.encoder_blocks:
            x = block(x)
            skips.append(x)

        # Decoder blocks
        for i, block in enumerate(self.decoder_blocks):
            if i == 0:
                skips.pop()
                x = block(x)
            else:
                x = torch.cat([x, skips.pop()], dim=1)
                x = block(x)

        # Output block
        x = self.output_block(x)

        # Return
        return x
