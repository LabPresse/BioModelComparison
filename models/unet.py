
# Import libraries
import torch
import torch.nn as nn


# Define the UNet class
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, n_features=8, n_blocks=3, n_layers_per_block=2, activation=nn.GELU()):
        super(UNet, self).__init__()

        # Set up attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block


        ### SET UP BLOCKS ###

        # Set up input block
        self.input_block = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, n_features, kernel_size=3, padding=1),
        )

        # Set up encoder blocks
        self.encoder_blocks = nn.ModuleList()
        for i in range(n_blocks):

            # Initialize layers
            layers = []

            # Downsample block
            n_in = n_features
            n_out = n_features
            layers.append(
                nn.Sequential(
                    nn.Conv2d(n_in, n_out, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(1, n_out),
                    activation,
                )
            )

            # Mixing layers
            for j in range(n_layers_per_block-1):
                n_in = n_features
                n_out = n_features
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(n_out, n_out, kernel_size=3, stride=1, padding=1),
                        nn.GroupNorm(1, n_out),
                        activation,
                    )
                )

            # Add to list
            self.encoder_blocks.append(nn.Sequential(*layers))

        
        # Set up decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(n_blocks):

            # Initialize layers
            layers = []

            # Upsample block
            n_in = n_features if i == 0 else n_features * 2
            n_out = n_features
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(n_in, n_out, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.GroupNorm(1, n_out),
                    activation,
                )
            )

            # Mixing layers
            for j in range(n_layers_per_block-1):
                n_in = n_features
                n_out = n_features
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(n_out, n_out, kernel_size=3, stride=1, padding=1),
                        nn.GroupNorm(1, n_out),
                        activation,
                    )
                )

            # Add to list
            self.decoder_blocks.append(nn.Sequential(*layers))

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
