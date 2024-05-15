
# Import libraries
import torch
import torch.nn as nn


# Define the UNet class
class UNet(nn.Module):
    def __init__(self, 
            in_channels, out_channels, 
            n_features=4, n_blocks=3, n_layers_per_block=2,
        ):
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
            nn.GroupNorm(1, in_channels, affine=False),  # Normalize input
            nn.Conv2d(in_channels, n_features, kernel_size=3, padding=1),
        )

        # Set up encoder blocks
        self.encoder_blocks = nn.ModuleList()
        n_in = n_features
        for i in range(n_blocks):
            n_out = n_in * 2

            # Initialize layers
            layers = []

            # Downsample block
            layers.append(
                nn.Sequential(
                    nn.Conv2d(n_in, n_out, kernel_size=3, stride=2, padding=1),
                    nn.InstanceNorm2d(n_out),
                    nn.ReLU(),
                )
            )

            # Mixing layers
            for j in range(n_layers_per_block-1):
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(n_out, n_out, kernel_size=3, stride=1, padding=1),
                        nn.InstanceNorm2d(n_out),
                        nn.ReLU(),
                    )
                )

            # Add to list
            self.encoder_blocks.append(nn.Sequential(*layers))

            # Update n_in
            n_in = n_out

        
        # Set up decoder blocks
        n_out = n_features * 2 ** n_blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(n_blocks):
            n_out = n_out // 2
            n_in = n_out * 2 if i == 0 else n_out * 4

            # Initialize layers
            layers = []

            # Upsample block
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(n_in, n_out, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.InstanceNorm2d(n_out),
                    nn.ReLU(),
                )
            )

            # Mixing layers
            for j in range(n_layers_per_block-1):
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(n_out, n_out, kernel_size=3, stride=1, padding=1),
                        nn.InstanceNorm2d(n_out),
                        nn.ReLU(),
                    )
                )

            # Add to list
            self.decoder_blocks.append(nn.Sequential(*layers))

            # Update n_in
            n_in = n_out

        # Set up output block
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
    
    # Set up model
    model = UNet(3, 1, n_features=8, n_blocks=3, n_layers_per_block=2)

    # Print number of parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Set up input tensor
    x = torch.randn(1, 3, 256, 256)

    # Test model
    y = model(x)

    # Print output shape
    print(y.shape)

