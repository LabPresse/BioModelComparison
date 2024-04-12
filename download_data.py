
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


# Get retina dataset
def get_retina_dataset():
    print('Getting retina dataset')

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


# Get fluorescent neurons dataset
def get_neurons_dataset():

    # Set paths
    path_data = os.path.join(root, 'neurons')
    
    # Download the dataset
    api.dataset_download_files(
        'nbroad/fluorescent-neuronal-cells',
        path=path_data,
        unzip=True,
    )

    # Move images to base directory
    oldpath = os.path.join(path_data, 'all_images', 'images')
    newpath = os.path.join(path_data, 'images')
    files = [f for f in os.listdir(oldpath) if f.lower().endswith('.png')]
    for file in files:
        os.rename(os.path.join(oldpath, file), os.path.join(newpath, file))

    # Move masks to base directory
    oldpath = os.path.join(path_data, 'all_masks', 'masks')
    newpath = os.path.join(path_data, 'masks')
    files = [f for f in os.listdir(oldpath) if f.lower().endswith('.png')]
    for file in files:
        os.rename(os.path.join(oldpath, file), os.path.join(newpath, file))

    # Remove old directories
    shutil.rmtree(os.path.join(path_data, 'all_images'))
    shutil.rmtree(os.path.join(path_data, 'all_masks'))

    # Done
    print('Finished downloading fluorescent neurons dataset')
    return


### Download datasets ###
if __name__ == "__main__":

    # Download datasets
    get_retina_dataset()
    get_bdello_datset()
    get_neurons_dataset()

    # Done
    print('All datasets downloaded successfully.')


