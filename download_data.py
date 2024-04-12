
# Import libraries
import os
import shutil
import rarfile
import requests
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate the API
api = KaggleApi()
api.authenticate()

# Set base data directory
root = './data'


# Get retina dataset FIVES
def get_retina_dataset_FIVES():
    print('Getting retina dataset FIVES')

    # Set paths
    url = 'https://figshare.com/ndownloader/files/34969398'
    path_data = os.path.join(root, 'retinas_FIVES')
    path_data_images = os.path.join(path_data, 'images')
    path_data_masks = os.path.join(path_data, 'masks')
    download_path = os.path.join(path_data, 'FIVES.rar')

    # Download the file
    print("Starting download...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(download_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
        print("Download completed successfully.")
    else:
        print("Failed to download the file. Status code:", response.status_code)
        return

    # Extract the file
    try:
        print("Starting extraction...")
        with rarfile.RarFile(download_path) as rf:
            rf.extractall(path_data)
        print("Extraction completed successfully.")
    except rarfile.Error as e:
        print("Failed to extract the file:", e)
    
    # Remove rar file
    os.remove(download_path)

    # Move images to base directory
    default_path = os.path.join(path_data, 'FIVES A Fundus Image Dataset for AI-based Vessel Segmentation')
    i = 0
    for folder in ['train', 'test']:
        files = os.listdir(os.path.join(default_path, folder, 'Original'))
        files = [f for f in files if f.lower().endswith('.png')]
        for file in files:
            os.rename(
                os.path.join(default_path, folder, 'Original', file), 
                os.path.join(path_data_images, f'img{i}.png')
            )
            os.rename(
                os.path.join(default_path, folder, 'Ground truth', file), 
                os.path.join(path_data_masks, f'img{i}.png')
            )
            i += 1

    # Remove default_path
    shutil.rmtree(default_path)
    
    # Done
    print('Finished downloading retina dataset FIVES')
    return


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

    # Set paths
    path_data = os.path.join(root, 'retinas_RFMiD')

    ### Download raw data ###
    print('Downloading raw data')

    # Download and unzip the dataset to the specified directory
    api.dataset_download_files(
        'andrewmvd/retinal-disease-classification', 
        path=path_data,
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
        files = os.listdir(os.path.join(path_data, set_path))
        files = [f for f in files if f.lower().endswith('.png')]
        for file in files:

            # Clean image and save
            image = clean_image(os.path.join(path_data, set_path, file))
            image.save(os.path.join(path_data, f'img{i}.png'))
            i += 1

    # Remove raw data
    shutil.rmtree(os.path.join(path_data, 'Training_Set'))
    shutil.rmtree(os.path.join(path_data, 'Test_Set'))
    shutil.rmtree(os.path.join(path_data, 'Evaluation_Set'))

    # Done
    print('Finished downloading retina dataset RFMiD')
    return


# Get bdello dataset
def get_bdello_datset():

    # Set paths
    path_data = os.path.join(root, 'bdello')

    # Download the dataset
    api.dataset_download_files(
        'shepbryan/phase-contrast-bdellovibrio',
        path=path_data,
        unzip=True
    )

    # Move images to base directory
    oldpath = os.path.join(path_data, 'Cleaned', 'images')
    newpath = os.path.join(path_data, 'images')
    files = [f for f in os.listdir(oldpath) if f.lower().endswith('.png')]
    for file in files:
        os.rename(os.path.join(oldpath, file), os.path.join(newpath, file))

    # Move masks to base directory
    oldpath = os.path.join(path_data, 'Cleaned', 'masks')
    newpath = os.path.join(path_data, 'masks')
    files = [f for f in os.listdir(oldpath) if f.lower().endswith('.png')]
    for file in files:
        os.rename(os.path.join(oldpath, file), os.path.join(newpath, file))

    # Remove old directories
    shutil.rmtree(os.path.join(path_data, 'Cleaned'))

    # Done
    print('Finished downloading bdello dataset')
    return


# Download datasets
# get_retina_dataset_RFMiD()
# get_retina_dataset_FIVES()
# get_bdello_datset()

# Done
print('All datasets downloaded successfully.')


