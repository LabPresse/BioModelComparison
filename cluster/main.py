
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


# Define training scheme function
def run_training_scheme(
        modelID, dataID, savename,
        ffcvid=0, img_size=256, n_epochs=50,
        verbose=True, plot=False,
        **kwargs
    ):
    """Run a training scheme on a model and dataset with given options."""

    # Print status
    if verbose:
        print(f'Starting training scheme: {savename}')

    # Get dataset
    if dataID == 'bdello':
        dataset = BdelloDataset(crop=(img_size, img_size), scale=2)
        in_channels = 1
        out_channels = 2
    elif dataID == 'retinas':
        dataset = RetinaDataset(crop=(img_size, img_size), scale=2)
        in_channels = 3
        out_channels = 2
    elif dataID == 'neurons':
        dataset = NeuronsDataset(crop=(img_size, img_size), scale=2)
        in_channels = 3
        out_channels = 2
        
    # Get 5-fold cross-validation split from ffcvID
    splits = random_split(dataset, [len(dataset) // 5] * 4 + [len(dataset) - 4 * (len(dataset) // 5)])
    testID = ffcvid
    valID = (ffcvid + 1) % 5
    trainIDs = [i for i in range(5) if i not in [testID, valID]]
    dataset_test = Subset(dataset, indices=splits[testID].indices)
    dataset_val = Subset(dataset, indices=splits[valID].indices)
    dataset_train = Subset(dataset, indices=[idx for i in trainIDs for idx in splits[i].indices])
    datasets = (dataset_train, dataset_val, dataset_test)

    # Get model
    if modelID == 'conv':
        # ConvolutionalNet
        model = ConvolutionalNet(
            in_channels=in_channels, 
            out_channels=out_channels,
            **kwargs
        )
    elif modelID == 'unet':
        # UNet
        model = UNet(
            in_channels=in_channels, 
            out_channels=out_channels,
            **kwargs
        )
    elif modelID == 'resnet':
        # ResNet
        model = ResNet(
            in_channels=in_channels, 
            out_channels=out_channels,
            **kwargs
        )
    elif modelID == 'vit':
        # VisionTransformer
        model = VisionTransformer(
            img_size=img_size, 
            in_channels=in_channels, 
            out_channels=out_channels,
            **kwargs
        )
    elif modelID == 'vim':
        # VisionMamba
        model = VisionMamba(
            img_size=img_size, 
            in_channels=in_channels, 
            out_channels=out_channels,
            **kwargs
        )

    # Set up model
    model = model.to(device)

    # # Print model parameters
    # print(f'{savename}: {count_parameters(model)} parameters')
    # return

    # Train model
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


# Run training scheme
if __name__ == "__main__":
        
    # Set up datasets, models, and options
    datasets = ['neurons', 'bdello', 'retinas']
    model_options = [
        # ConvolutionalNet
        ['conv', {'n_layers': 8}],
        ['conv', {'n_layers': 16}],
        ['conv', {'n_layers': 32}],
        # UNet
        ['unet', {'n_blocks': 2}],
        ['unet', {'n_blocks': 3}],
        ['unet', {'n_blocks': 4}],
        # ResNet
        ['resnet', {'n_blocks': 2}],
        ['resnet', {'n_blocks': 3}],
        ['resnet', {'n_blocks': 4}],
        # VisionTransformer
        ['vit', {'img_size': 128, 'n_layers': 4}],
        ['vit', {'img_size': 128, 'n_layers': 8}],
        ['vit', {'img_size': 128, 'n_layers': 16}],
        # # VisionMamba
        # ['vim', {'img_size': 128, 'n_layers': 4}],
        # ['vim', {'img_size': 128, 'n_layers': 8}],
        # ['vim', {'img_size': 128, 'n_layers': 16}],
    ]

    # Set up all jobs
    all_jobs = []
    for ffcvid in range(5):
        for dataID in datasets:
            for modelID, options in model_options:
                all_jobs.append((
                    modelID, 
                    dataID, 
                    options,
                    ffcvid,
                ))
    n_jobs = len(all_jobs)

    # for jobID in range(n_jobs):
    #     modelID, dataID, options, ffcvid = all_jobs[jobID]
    #     savename = get_savename(modelID, dataID, options, ffcvid)
    #     run_training_scheme(
    #         modelID, 
    #         dataID, 
    #         savename,
    #         ffcvid=ffcvid,
    #         verbose=False,
    #         **options
    #     )
    
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

