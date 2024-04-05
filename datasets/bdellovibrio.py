
# Import libraries
import os
import PIL
import torch
import random
import numpy as np
from skimage.draw import disk
from skimage import filters
from torch.utils.data import Dataset


# Define BdellovibrioDataset class
class BdellovibrioDataset(Dataset):
    def __init__(self,  data_path, mask_path, dot_radius=5, image_shape=(128, 128), num_images=1000):
        super().__init__()

        # Set attributes
        self.img_size = image_shape[0]
        self.img_channels = 3
        self.target_channels = 1
        self.data_path = data_path
        self.mask_path = mask_path
        self.dot_radius = dot_radius
        self.image_shape = image_shape
        self.num_images = num_images

        # Get list of files
        masks = os.listdir(self.mask_path)
        files = os.listdir(self.data_path)
        files = [file for file in files if file.endswith('.png')]
        files = [file for file in files if file[:-4] + '_mask.png' in masks]
        self.files = files
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):

        # Load image and mask
        file = self.files[idx]
        image = PIL.Image.open(os.path.join(self.data_path, file))
        mask = PIL.Image.open(os.path.join(self.mask_path, file[:-4]+'_mask.png'))

        # Randomly crop image
        while True:
            row = np.random.randint(0, image.size[0] - self.image_shape[0])
            col = np.random.randint(0, image.size[1] - self.image_shape[1])
            image_crop = image.crop((row, col, row + self.image_shape[0], col + self.image_shape[1]))
            mask_crop = mask.crop((row, col, row + self.image_shape[0], col + self.image_shape[1]))
            if np.sum(np.array(mask_crop)) > 0:
                break
        image = image_crop
        mask = mask_crop

        # Randomly rotate and flip image
        rot = np.random.randint(0, 4)
        image = image.rotate(90 * rot)
        mask = mask.rotate(90 * rot)
        if np.random.rand() < .5:
            image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        # Normalize image
        image = np.array(image).astype(float)
        image -= np.mean(image)
        if np.std(image) > 0:
            image /= np.std(image)
        
        # Normalize mask
        mask = np.array(mask).astype(float)
        mask -= np.min(mask)
        if np.max(mask) > 0:
            mask /= np.max(mask)
        # Fill in mask
        mask = self.enlarge_blob(mask)

        # Convert to tensors
        image = torch.tensor(np.array(image), dtype=torch.float32)
        image = image.unsqueeze(0)
        mask = torch.tensor(np.array(mask), dtype=torch.float32)
        mask = mask.unsqueeze(0)

        # Return image
        return image, mask
    
    def enlarge_blob(self, mask, dot_radius=None):
        """Enlarge the blob in the mask."""

        # Set radius
        if dot_radius is None:
            dot_radius = self.dot_radius

        # Get blob
        blob = np.where(mask > 0)
        blob = np.array([blob[0], blob[1]]).T

        # Get random point
        for point in blob:
            # Enlarge blob
            rr, cc = disk(point, dot_radius, shape=mask.shape)
            mask[rr, cc] = 1

        # Return mask
        return mask


if __name__ == '__main__':
    
    # Import libraries
    import torch
    import matplotlib.pyplot as plt

    # Set up figure
    fig, ax = plt.subplots(1, 2)
    plt.ion()
    plt.show()

    # Define dataset
    data_path = os.path.join(os.environ['DATAPATH'], 'Bdello/Images/')
    mask_path = os.path.join(os.environ['DATAPATH'], 'Bdello/Masks/')
    dataset = BBImages(data_path=data_path, mask_path=mask_path)

    # Plot samples
    for i in range(5):
        image, mask = dataset[i]
        ax[0].cla()
        ax[0].imshow(image[0, :, :], cmap='gray')
        ax[1].cla()
        ax[1].imshow(mask[0, :, :], cmap='gray')
        plt.pause(.25)