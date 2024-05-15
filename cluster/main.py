
# Import libraries
import os
import sys
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Subset, random_split

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from helper_functions import plot_images, count_parameters, check_gradient, convert_to_serializable, get_savename
from models.conv import ConvolutionalNet
from models.unet import UNet
from models.resnet import ResNet
from models.vit import VisionTransformer
from models.vim import VisionMamba
# from models.resnet import ResNet
from datasets.bdello import BdelloDataset
from datasets.retinas import RetinaDataset
from datasets.neurons import NeuronsDataset
from training import train_model
from testing import evaluate_model


# Set environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outfiles')


# Define get model function
def get_model(modelID, img_size=None, scale=None, **kwargs):
    """Get a model with given options."""
    if modelID == 'conv':
        # ConvolutionalNet
        model = ConvolutionalNet(**kwargs)
    elif modelID == 'unet':
        # UNet
        model = UNet(**kwargs)
    elif modelID == 'resnet':
        # ResNet
        model = ResNet(**kwargs)
    elif modelID == 'vit':
        # VisionTransformer
        model = VisionTransformer(img_size=img_size, **kwargs)
    elif modelID == 'vim':
        # VisionMamba
        model = VisionMamba(img_size=img_size, **kwargs)
    return model


# Define training scheme function
def run_training_scheme(
        modelID, dataID, savename,
        img_size=256, scale=2, ffcvid=0, n_epochs=20,
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
    if dataID == 'bdello':
        dataset = BdelloDataset(crop=(img_size, img_size), scale=scale)
        in_channels = 1
        out_channels = 2
    elif dataID == 'retinas':
        dataset = RetinaDataset(crop=(img_size, img_size), scale=scale)
        in_channels = 3
        out_channels = 2
    elif dataID == 'neurons':
        dataset = NeuronsDataset(crop=(img_size, img_size), scale=scale)
        in_channels = 3
        out_channels = 2
        
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

    # Set up model
    if verbose:
        print('Setting up model.')
    model = get_model(
        modelID,
        img_size=img_size, 
        in_channels=in_channels,
        out_channels=out_channels,
        **kwargs
    )
    model = model.to(device)

    # Train model
    if verbose:
        print('Training model.')
    model, statistics = train_model(
        model, datasets, os.path.join(outpath, f'{savename}.pth'),
        verbose=verbose, plot=plot,
        n_epochs=n_epochs,
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


# Define look at jobs function for debugging
def look_at_jobs(jobs):
    """Print out information about jobs."""

    # Loop over jobs
    for jobID in range(len(jobs)):

        # Get job parameters
        modelID, dataID, options, ffcvid = jobs[jobID]
        img_size = options.get('img_size', 256)
        scale = options.get('scale', 2)

        # Get dataset
        if dataID == 'bdello':
            dataset = BdelloDataset(crop=(img_size, img_size), scale=scale)
            in_channels = 1
            out_channels = 2
        elif dataID == 'retinas':
            dataset = RetinaDataset(crop=(img_size, img_size), scale=scale)
            in_channels = 3
            out_channels = 2
        elif dataID == 'neurons':
            dataset = NeuronsDataset(crop=(img_size, img_size), scale=scale)
            in_channels = 3
            out_channels = 2
        
        # Get save name
        savename = get_savename(modelID, dataID, options, ffcvid)

        # Get model
        model = get_model(
            modelID, 
            in_channels=in_channels,
            out_channels=out_channels,
            **options
        )

        # Print number of parameters
        print(f'{savename}\n--{len(dataset)}\n--{count_parameters(model)}')
        # print(f'{int(.6 * len(dataset)) // 32}')
    
    # Done
    return

# Run training scheme
if __name__ == "__main__":
        
    # Set up datasets, models, and options
    datasets = ['retinas', 'neurons', 'bdello', ]
    model_options = [
        # ConvolutionalNet
        ['conv', {'n_layers': 4}],
        ['conv', {'n_layers': 8}],
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
        ['vit', {'n_layers': 8}],
        ['vit', {'n_layers': 16}],
        # # VisionMamba
        # ['vim', {'n_layers': 4}],
        # ['vim', {'n_layers': 8}],
        # ['vim', {'n_layers': 16}],
    ]

    # Set up all jobs
    all_jobs = []
    for ffcvid in range(5):                         # 5-fold cross-validation
        for dataID in datasets:                     # Datasets
            for modelID, options in model_options:  # Models

                # Set scale
                if dataID == 'retinas':
                    scale = 2
                elif dataID == 'bdello':
                    scale = 2
                elif dataID == 'neurons':
                    scale = 1
                    
                # Set image size
                if modelID == 'vit' or modelID == 'vim':
                    # Use smaller image for memory-intensive models
                    img_size = 128
                    scale *= 2
                else:
                    img_size = 256

                # Update options
                options = {**options, 'img_size': img_size, 'scale': scale}

                # Add job
                all_jobs.append((modelID, dataID, options, ffcvid))

    # # Look at jobs
    # look_at_jobs(all_jobs)
    
    # Get job id from sys
    jobID = 0
    if len(sys.argv) > 1:
        jobID = int(sys.argv[1])

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

