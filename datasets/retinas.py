

# Import libraries
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


# Create a Retina vessel datset class
class RetinaDataset(Dataset):
    def __init__(self, crop=None, scale=1):
        super(RetinaDataset, self).__init__()

        # Set up attributes
        self.crop = crop
        self.scale = scale
        self.base_shape = (2048, 2048)

        # Calculate constants
        if crop is None:
            self.crops_per_image = 1
        else:
            self.crops_per_image = (
                (self.base_shape[0] // scale // crop[0])
                * (self.base_shape[1] // scale // crop[1])  
            )

        # Set up root directory
        self.root = os.path.join('data/retinas')

        # Get files
        self.files = os.listdir(os.path.join(self.root, 'images'))
        self.files = [f for f in self.files if f.endswith('.png')]

    def __len__(self):
        return len(self.files) * self.crops_per_image
    
    def __getitem__(self, idx):

        # Get file ID and crop ID
        file_id = idx // self.crops_per_image
        crop_id = idx % self.crops_per_image

        # Get image amd mask
        file = self.files[file_id]
        image = Image.open(os.path.join(self.root, 'images', file))
        image = transforms.ToTensor()(image)
        mask = Image.open(os.path.join(self.root, 'masks', file))
        mask = transforms.ToTensor()(mask)

        # Configure mask
        mask = mask[0, :, :] > 0.5
        mask = mask.long()

        # Scale
        scale = self.scale
        image = image[:, ::scale, ::scale]
        mask = mask[::scale, ::scale]

        # Crop
        crop = self.crop
        img_shape = image.shape[1:]
        if crop is not None:
            row = crop_id // (img_shape[1] // crop[1])
            col = crop_id % (img_shape[1] // crop[1])
            image = image[:, row*crop[0]:(row+1)*crop[0], col*crop[1]:(col+1)*crop[1]]
            mask = mask[row*crop[0]:(row+1)*crop[0], col*crop[1]:(col+1)*crop[1]]

        # Return
        return image, mask


# Test
if __name__ == "__main__":

    # Create a dataset object
    dataset = RetinaDataset()

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

