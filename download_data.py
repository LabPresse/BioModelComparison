
# Import libraries
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate the API
api = KaggleApi()
api.authenticate()

# Set base data directory
root = './data'


# Get retina dataset RFMiD
def get_retina_dataset_RFMiD():
    print('Getting retina dataset RFMiD')

    # Import libraries
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision import transforms
    from torch.utils.data import Dataset
    from scipy.ndimage import label, binary_erosion


    ### Set up directories ###
    print('Setting up directories')

    # Create target directory
    path_raw = os.path.join(root, 'retinas_RFMiD')
    path_cleaned = os.path.join(root, 'retinas_RFMiD_cleaned')
    if not os.path.exists(path_raw):
        os.makedirs(path_raw)
    if not os.path.exists(path_cleaned):
        os.makedirs(path_cleaned)


    ### Download raw data ###
    print('Downloading raw data')

    # Download and unzip the dataset to the specified directory
    api.dataset_download_files(
        'andrewmvd/retinal-disease-classification', 
        path=path_raw,
        unzip=True
    )


    ### Clean images ###
    print('Cleaning images')

    # Define image cleaner function
    def clean_image(filepath, threshold=10/255):

        # Load image
        image = Image.open(filepath)
        image = transforms.ToTensor()(image)
        
        ### GET IMAGE REGION ###
        # Convert to numpy
        mask = image.cpu().detach().numpy()  # mask will be the not-black region of the image
        # Threshold image
        mask = mask[0, :, :] > threshold
        # Label connected components
        mask, n_labels = label(mask)
        label_counts = [(mask == i).sum() for i in range(1, n_labels+1)]
        largest_label = label_counts.index(max(label_counts)) + 1
        # Set mask to largest component
        mask = (mask == largest_label)
        # Set boundary pixels of component to zero
        mask = binary_erosion(mask, iterations=1)
        # Set mask to tensor
        mask = torch.tensor(mask).float()

        # Get x and y range of mask
        x_min = mask.any(axis=0).numpy().argmax()
        x_max = mask.shape[1] - mask.any(axis=0).numpy()[::-1].argmax()
        y_min = mask.any(axis=1).numpy().argmax()
        y_max = mask.shape[0] - mask.any(axis=1).numpy()[::-1].argmax()

        # Crop
        image = image[:, y_min:y_max, x_min:x_max]
        mask = mask[y_min:y_max, x_min:x_max]

        # Pad to square
        max_side = max(image.shape[1], image.shape[2])
        pad_left = (max_side - image.shape[2]) // 2
        pad_right = max_side - image.shape[2] - pad_left
        pad_top = (max_side - image.shape[1]) // 2
        pad_bottom = max_side - image.shape[1] - pad_top
        image = torch.nn.functional.pad(image, (pad_left, pad_right, pad_top, pad_bottom))
        mask = torch.nn.functional.pad(mask, (pad_left, pad_right, pad_top, pad_bottom))

        # Resize to 2048x2048
        image = torch.nn.functional.interpolate(image.unsqueeze(0), (2048, 2048)).squeeze(0)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), (2048, 2048)).squeeze(0).squeeze(0) > 0

        # Set background to zero
        image = image * mask

        # Convert to PIL
        image = transforms.ToPILImage()(image)

        # Return image
        return image

    # Loop over cases
    i = 0
    for set_path in [
        'Training_Set/Training_Set/Training', 
        'Test_Set/Test_Set/Test', 
        'Evaluation_Set/Evaluation_Set/Validation'
        ]:
        print(f'-- Cleaning set: {set_path}')

        # Loop over images
        files = os.listdir(os.path.join(path_raw, set_path))
        files = [f for f in files if f.lower().endswith('.png')]
        for file in files:

            # Clean image and save
            image = clean_image(os.path.join(path_raw, set_path, file))
            image.save(os.path.join(path_cleaned, f'img{i}.png'))
            i += 1

    # Done
    print('Finished downloading retina dataset RFMiD')
    return


# Download datasets
get_retina_dataset_RFMiD()


