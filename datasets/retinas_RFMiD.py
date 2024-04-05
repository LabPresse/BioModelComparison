

# Import libraries
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


# Create a Retina RFMiD datset class
class RetinaRFMiDDataset(Dataset):

    def __init__(self, crop=(256, 256), scale=1):
        super(RetinaRFMiDDataset, self).__init__()

        # Set up attributes
        self.crop = crop
        self.scale = scale
        self.base_shape = (2048, 2048)

        # # Calculate constants
        # self.crops_per_image = (self.base_shape[0] // crop[0]) * (self.base_shape[1] // crop[1])

        # Set up root directory
        self.root = os.path.join('data/retinas_RFMiD_cleaned')

        # Get files
        self.files = os.listdir(self.root)
        self.files = [f for f in self.files if f.endswith('.png')]

    def __len__(self):
        # return len(self.files) * self.crops_per_image
        return len(self.files)
    
    def __getitem__(self, idx):

        # # Get file ID and crop ID
        # file_id = idx // self.crops_per_image
        # crop_id = idx % self.crops_per_image

        # Get file
        file = self.files[idx]

        # Get image
        image = Image.open(os.path.join(self.root, file))
        image = transforms.ToTensor()(image)

        # Scale
        scale = self.scale
        image = image[:, ::scale, ::scale]

        # Crop
        crop = self.crop
        img_shape = image.shape[1:]
        if crop is not None:
            # row = crop_id // (base_shape[1] // crop[1])
            # col = crop_id % (base_shape[1] // crop[1])
            # image = image[:, row*crop[0]:(row+crop[0]), col*crop[1]:(col+1)*crop[1]]
            row = torch.randint(0, img_shape[0] - crop[0], (1,)).item()
            col = torch.randint(0, img_shape[1] - crop[1], (1,)).item()
            image = image[:, row:row+crop[0], col:col+crop[1]]

        # Return
        return image


# Test
if __name__ == "__main__":

    # Create a dataset object
    dataset = RetinaRFMiDDataset()

    # Set up a figure
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    plt.ion()
    plt.show()

    # Loop
    for i in range(len(dataset)):

        # Get item
        image = dataset[i]

        # Show
        ax.cla()
        ax.imshow(image.permute(1, 2, 0))
        plt.pause(0.1)

    # Done
    print('Done')

