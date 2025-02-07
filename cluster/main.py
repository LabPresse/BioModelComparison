
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
        from data.retinas import RetinaDataset
        dataset = RetinaDataset(crop=(128, 128), scale=4)
        in_channels = 3
        out_channels = 2
    elif dataID == 'neurons':
        from data.neurons import NeuronsDataset
        dataset = NeuronsDataset(crop=(128, 128), scale=8)
        in_channels = 3
        out_channels = 2
    elif dataID == 'bdello':
        from data.bdello import BdelloDataset
        dataset = BdelloDataset(crop=(128, 128), scale=4)
        in_channels = 1
        out_channels = 2
    elif dataID == 'letters':
        from data.letters import LettersDataset
        dataset = LettersDataset(shape=(128, 128), sigma=0.1)
        in_channels = 1
        out_channels = 2
    elif 'letters' in dataID:
        from data.letters import LettersDataset
        options = {}
        if 'blur' in dataID:
            options['blur'] = float(dataID.split('blur=')[1].split('_')[0])
        if 'sigma' in dataID:
            options['sigma'] = float(dataID.split('sigma=')[1].split('_')[0])
        dataset = LettersDataset(shape=(128, 128), **options)
        in_channels = 1
        out_channels = 2

    elif dataID == 'testing':  # Small dataset for debugging and testing
        from data.letters import ChineseCharacters
        dataset = ChineseCharacters(shape=(128, 128), sigma=0.25, max_characters=100)
        in_channels = 1
        out_channels = 2

    # Get model
    if verbose:
        print('Setting up model.')
    if modelID == 'conv':
        from models.conv import ConvolutionalNet
        model = ConvolutionalNet(
            in_channels=in_channels, out_channels=out_channels, **kwargs
        )
    elif modelID == 'unet':
        from models.unet import UNet
        model = UNet(
            in_channels=in_channels, out_channels=out_channels, **kwargs
        )
    elif modelID == 'resnet':
        from models.resnet import ResNet
        model = ResNet(
            in_channels=in_channels, out_channels=out_channels, **kwargs
        )
    elif modelID == 'vit':
        from models.vit import VisionTransformer
        model = VisionTransformer(
            img_size=128, in_channels=in_channels, out_channels=out_channels, **kwargs
        )
    elif modelID == 'vim':
        from models.vim import VisionMamba
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
    # datasets = ['neurons', 'retinas', 'letters', 'bdello']
    datasets = []
    for blur in [2, 4, 8, 16, 32]:
        for sigma in [.25, .5, 1, 2, 4]:
            datasets.append(f'letters_blur={blur}_sigma={sigma}')
    model_options = [
        # ConvolutionalNet
        ['conv', {'n_layers': 8}],
        ['conv', {'n_layers': 12}],
        ['conv', {'n_layers': 16}],
        # # UNet
        ['unet', {'n_blocks': 2}],
        ['unet', {'n_blocks': 3}],
        ['unet', {'n_blocks': 4}],
        # # VisionTransformer
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
    for ffcvid in range(5):                         # 5-fold cross-validation 
        for dataID in datasets:                     # Datasets
            for modelID, options in model_options:  # Models
                all_jobs.append((modelID, dataID, options, ffcvid))

    # Filter jobs that are already complete
    all_jobs_filtered = []
    for modelID, dataID, options, ffcvid in all_jobs:
        savename = get_savename(modelID, dataID, options, ffcvid)
        if f'{savename}.json' not in os.listdir(outpath):
            all_jobs_filtered.append((modelID, dataID, options, ffcvid))
    print(f'Running {len(all_jobs_filtered)} jobs out of {len(all_jobs)} total.')
    all_jobs = all_jobs_filtered
    
    # Get job id from sys
    jobID = 0
    if len(sys.argv) > 1:
        jobID = int(sys.argv[1])

    # Loop through all jobs
    for jobID in range(len(all_jobs)):

        # Get job parameters
        modelID, dataID, options, ffcvid = all_jobs[jobID]
        savename = get_savename(modelID, dataID, options, ffcvid)
        if dataID == 'neurons' and (modelID == 'conv' or modelID == 'unet'):
            n_epochs = 500
        elif dataID == 'bdello' and (modelID == 'vit'):
            n_epochs = 500
        else:
            n_epochs = 100
    
        # Run training scheme
        run_training_scheme(
            modelID, 
            dataID, 
            savename,
            n_epochs=100,
            **options
        )
    
    # Done
    print('Done.')

