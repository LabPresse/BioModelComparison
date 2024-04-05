
# Import libraries
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Subset, random_split

# Import local modules
from helper_functions import plot_images, count_parameters, check_gradient
from models.unet import UNet
from models.vit import VisionTransformer
from models.conv import ConvolutionalNet
# from models.vae import VariationalAutoencoder
# from models.resnet import ResNet
from training import train_model
# from datasets.bdellovibrio import BdellovibrioDataset
from datasets.retina_FIVES import RetinaDatasetFives

# Set environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define training scheme function
def run_training_scheme(
        modelID, dataID, 
        ffcvID=0, img_size=128, pretrain_ae=False, 
        trainargs=None, pretrainargs=None, **modelargs
    ):
    """Run a training scheme on a model and dataset with given options."""

    # Set up inputs
    if trainargs is None:
        trainargs = {}
    if pretrainargs is None:
        pretrainargs = {}

    # Get dataset
    if dataID == 'bdellovibrio':
        dataset = BdellovibrioDataset(image_shape=(img_size, img_size))
        in_channels = 3
        out_channels = 1
    elif dataID == 'retina_FIVES':
        dataset = RetinaDatasetFives(crop=(img_size, img_size))
        in_channels = 3
        out_channels = 1
        
    # Get 5-fold cross-validation split from ffcvID
    splits = random_split(dataset, [len(dataset) // 5] * 5)
    testID = ffcvID
    valID = (ffcvID + 1) % 5
    trainIDs = [i for i in range(5) if i not in [testID, valID]]
    dataset_test = Subset(dataset, indices=splits[testID].indices)
    dataset_val = Subset(dataset, indices=splits[valID].indices)
    dataset_train = Subset(dataset, indices=[idx for i in trainIDs for idx in splits[i].indices])

    # Get model
    if modelID == 'vit':
        model = VisionTransformer(
            img_size=img_size, 
            in_channels=in_channels, 
            out_channels=out_channels,
            **modelargs
        )
        pretrainargs['mae'] = True
    elif modelID == 'unet':
        model = UNet(
            in_channels=in_channels, 
            out_channels=out_channels,
            **modelargs
        )
    elif modelID == 'conv':
        model = ConvolutionalNet(
            in_channels=in_channels, 
            out_channels=out_channels,
            **modelargs
        )

    # Pretrain as autoencoder if necessary
    if pretrain_ae:
        model.set_output_layer(in_channels)
        model = train_model(
            model, dataset_train, dataset_val, 'outfiles/model_ae.pth', 
            segmentation=False, autoencoder=True,
            verbose=True, plot=True, device=device,
            **pretrainargs
        )
        model.reset_output_layer(out_channels)

    # Train model
    model = train_model(
        model, dataset_train, dataset_val, 'outfiles/model.pth', 
        segmentation=True,
        verbose=True, plot=True, device=device,
        **trainargs
    )

    # Done
    print('Done.')
    return


# Run training scheme
if __name__ == "__main__":
    run_training_scheme(
        'vit', 'retina_FIVES', pretrain_ae=True,
    )

