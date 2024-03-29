

# Import libraries
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


# Create a Retina vessel datset class
class RetinaVesselDataset(Dataset):

    def __init__(self, crop=(256, 256), scale=1):

        # Set up attributes
        self.crop = crop
        self.scale = scale
        self.base_shape = (2048, 2048)

        # Set up root directory
        self.root = os.path.join(
            os.environ['DATAPATH'],
            'Retinas',
            'FIVES A Fundus Image Dataset for AI-based Vessel Segmentation',
            'train',
        )

        # Get files
        self.files = os.listdir(os.path.join(self.root, 'Original'))
        self.files = [f for f in self.files if f.endswith('.png')]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):

        # Get file
        file = self.files[idx]

        # Get image
        image = Image.open(os.path.join(self.root, 'Original', file))
        image = transforms.ToTensor()(image)

        # Get mask
        mask = Image.open(os.path.join(self.root, 'Ground truth', file))
        mask = transforms.ToTensor()(mask)
        mask = mask[0, :, :] > 0.5
        mask = mask.long()

        # Scale
        scale = self.scale
        image = image[:, ::scale, ::scale]
        mask = mask[::scale, ::scale]

        # Crop
        crop = self.crop
        base_shape = self.base_shape
        if crop is not None:
            x = torch.randint(0, base_shape[0] - crop[0], (1,)).item()
            y = torch.randint(0, base_shape[1] - crop[1], (1,)).item()
            image = image[:, x:x+crop[0], y:y+crop[1]]
            mask = mask[x:x+crop[0], y:y+crop[1]]

        # Return
        return image, mask


# Test
if __name__ == "__main__":

    # Create a dataset object
    dataset = RetinaVesselDataset()

    # Set up a figure
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2)
    plt.ion()
    plt.show()

    # Loop
    for i in range(len(dataset)):

        # Get item
        image, mask = dataset[i]

        # Show
        ax[0].cla()
        ax[1].cla()
        ax[0].imshow(image.permute(1, 2, 0))
        ax[1].imshow(mask)
        plt.pause(0.1)

    # Done
    print('Done')

