
# Import libraries
print('Importing libraries')
import os
import sys
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Subset, random_split

# Import local modules
print('Importing local modules')
sys.path.append( # Add parent directory to path
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
from models.conv import ConvolutionalNet
from models.unet import UNet
from models.resnet import ResNet
from models.vit import VisionTransformer
from models.vim import VisionMamba
from data.bdello import BdelloDataset
from data.neurons import NeuronsDataset
from data.retinas import RetinaDataset
from training import train_model
from testing import evaluate_model
from helper_functions import (
    plot_images, count_parameters, check_gradient, convert_to_serializable, get_savename
)

# Set environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outfiles')


# Define training scheme function
def run_training_scheme(
        modelID, dataID, savename, 
        ffcvid=0, n_epochs=100,
        verbose=True, plot=False,
        **kwargs
    ):
    """Run a training scheme on a model and dataset with given options."""

    # Print status
    if verbose:
        print(f'Starting training scheme: {savename}')

    # Get dataset
    if verbose:
        print('Loading dataset.')
    if dataID == 'retinas':
        dataset = RetinaDataset(crop=(128, 128), scale=4)
        in_channels = 3
        out_channels = 2
    elif dataID == 'neurons':
        dataset = NeuronsDataset(crop=(128, 128), scale=2)
        in_channels = 3
        out_channels = 2
    elif dataID == 'bdello':
        dataset = BdelloDataset(crop=(128, 128), scale=4)
        in_channels = 1
        out_channels = 2

    # Get model
    if verbose:
        print('Setting up model.')
    if modelID == 'conv':
        model = ConvolutionalNet(
            in_channels=in_channels, out_channels=out_channels, **kwargs
        )
    elif modelID == 'unet':
        model = UNet(
            in_channels=in_channels, out_channels=out_channels, **kwargs
        )
    elif modelID == 'resnet':
        model = ResNet(
            in_channels=in_channels, out_channels=out_channels, **kwargs
        )
    elif modelID == 'vit':
        model = VisionTransformer(
            img_size=128, in_channels=in_channels, out_channels=out_channels, **kwargs
        )
    elif modelID == 'vim':
        model = VisionMamba(
            img_size=128, in_channels=in_channels, out_channels=out_channels, **kwargs
        )

    # Move model to device
    model = model.to(device)
        
    # Get 5-fold cross-validation split from ffcvID
    if verbose:
        print('Splitting dataset.')
    splits = random_split(dataset, [len(dataset) // 5] * 4 + [len(dataset) - 4 * (len(dataset) // 5)])
    testID = ffcvid
    valID = (ffcvid + 1) % 5
    trainIDs = [i for i in range(5) if i not in [testID, valID]]
    dataset_test = Subset(dataset, indices=splits[testID].indices)
    dataset_val = Subset(dataset, indices=splits[valID].indices)
    dataset_train = Subset(dataset, indices=[idx for i in trainIDs for idx in splits[i].indices])
    datasets = (dataset_train, dataset_val, dataset_test)

    # Train model
    if verbose:
        print('Training model.')
    model, statistics = train_model(
        model, datasets, os.path.join(outpath, f'{savename}.pth'),
        n_epochs=n_epochs,
        verbose=verbose, plot=plot,
    )

    # Test model
    if verbose:
        print('Evaluating model.')
    test_metrics = evaluate_model(model, dataset_test, verbose=verbose, plot=plot)
    statistics['test_metrics'] = test_metrics

    # Save statistics as json
    if verbose:
        print('Saving statistics.')
    with open(os.path.join(outpath, f'{savename}.json'), 'w') as f:
        json.dump(statistics, f, default=convert_to_serializable)

    # Done
    if verbose:
        print('Finished training scheme.')
    return


# Run training scheme
if __name__ == "__main__":
        
    # Set up datasets, models, and options
    datasets = ['retinas', 'neurons', 'bdello', ]
    model_options = [
        # ConvolutionalNet
        ['conv', {'n_layers': 8}],
        ['conv', {'n_layers': 12}],
        ['conv', {'n_layers': 16}],
        # UNet
        ['unet', {'n_blocks': 2}],
        ['unet', {'n_blocks': 3}],
        ['unet', {'n_blocks': 4}],
        # ResNet
        ['resnet', {'n_blocks': 2}],
        ['resnet', {'n_blocks': 3}],
        ['resnet', {'n_blocks': 4}],
        # VisionTransformer
        ['vit', {'n_layers': 4}],
        ['vit', {'n_layers': 6}],
        ['vit', {'n_layers': 8}],
        # # VisionMamba
        ['vim', {'n_layers': 4}],
        ['vim', {'n_layers': 6}],
        ['vim', {'n_layers': 8}],
    ]

    # Set up all jobs
    all_jobs = []
    for ffcvid in range(1):                         # 5-fold cross-validation TODO: Change to 5
        for dataID in datasets:                     # Datasets
            for modelID, options in model_options:  # Models
                all_jobs.append((modelID, dataID, options, ffcvid))
    
    # # Get job id from sys
    # jobID = 0
    # if len(sys.argv) > 1:
    #     jobID = int(sys.argv[1])

    # Loop over jobIDs
    for jobID in range(10, 15):  # These timed out on the cluster

        # Get job parameters
        modelID, dataID, options, ffcvid = all_jobs[jobID]
        savename = get_savename(modelID, dataID, options, ffcvid)
        
        # Run training scheme
        run_training_scheme(
            modelID, 
            dataID, 
            savename,
            verbose=True,
            **options
        )

    # Done
    print('Done.')

