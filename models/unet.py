
# Import libraries
import torch
import torch.nn as nn


# Define the UNet class
class UNet(nn.Module):
    def __init__(self, 
            in_channels, out_channels, 
            n_features=8, n_blocks=3, n_layers_per_block=2, expansion=2, activation=None
        ):
        super(UNet, self).__init__()

        # Set up attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.n_layers_per_block = n_layers_per_block
        self.activation = activation

        # Get activation function
        if activation is None:
            activation_layer = nn.ReLU()
        elif activation.lower() == 'relu':
            activation_layer = nn.ReLU()
        elif activation.lower() == 'leakyrelu':
            activation_layer = nn.LeakyReLU()
        elif activation.lower() == 'gelu':
            activation_layer = nn.GELU()


        ### SET UP BLOCKS ###

        # Set up input block
        self.input_block = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            nn.Conv2d(in_channels, n_features, kernel_size=3, padding=1),
        )

        # Set up encoder blocks
        self.encoder_blocks = nn.ModuleList()
        n_in = n_features
        for i in range(n_blocks):

            # Initialize layers
            layers = []

            # Downsample block
            n_out = n_in * expansion
            layers.append(
                nn.Sequential(
                    nn.Conv2d(n_in, n_out, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(1, n_out),
                    activation_layer,
                )
            )

            # Mixing layers
            for j in range(n_layers_per_block-1):
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(n_out, n_out, kernel_size=3, stride=1, padding=1),
                        nn.GroupNorm(1, n_out),
                        activation_layer,
                    )
                )

            # Add to list
            self.encoder_blocks.append(nn.Sequential(*layers))

            # Update n_in
            n_in = n_out

        
        # Set up decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(n_blocks):

            # Initialize layers
            layers = []

            # Upsample block
            n_out = n_in // expansion
            n_in = n_in if i == 0 else n_in * 2
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(n_in, n_out, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.GroupNorm(1, n_out),
                    activation_layer,
                )
            )

            # Mixing layers
            for j in range(n_layers_per_block-1):
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(n_out, n_out, kernel_size=3, stride=1, padding=1),
                        nn.GroupNorm(1, n_out),
                        activation_layer,
                    )
                )

            # Add to list
            self.decoder_blocks.append(nn.Sequential(*layers))

            # Update n_in
            n_in = n_out

        # Set up output block
        self.set_output_layer(out_channels)

    def set_output_layer(self, out_channels):
        """Set the output layer to have the given number of channels."""
        self.output_block = nn.Sequential(
            nn.Conv2d(self.n_features, out_channels, kernel_size=1),
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

# Test model
if __name__ == '__main__':
    model = UNet(3, 1, n_features=8, n_blocks=3, n_layers_per_block=2)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
    print(y.min(), y.max())