
# Import libraries
import os
import sys
import json
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Subset, random_split

# Import local modules
from helper_functions import plot_images, count_parameters, check_gradient, convert_to_serializable
from models.conv import ConvolutionalNet
from models.unet import UNet
from models.resnet import ResNet
from models.vit import VisionTransformer
# from models.resnet import ResNet
from datasets.bdello import BdelloDataset
from datasets.retinas import RetinaDataset
from datasets.neurons import NeuronsDataset
from training import train_model
from testing import evaluate_model


# Set environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
outpath = 'outfiles'


# Define training scheme function
def run_training_scheme(
        modelID, dataID, savename,
        ffcvID=0, img_size=256, pretrain=False, n_epochs=50, max_samples=None,
        verbose=True, plot=False,
        trainargs=None, pretrainargs=None, **modelargs
    ):
    """Run a training scheme on a model and dataset with given options."""

    # Print status
    if verbose:
        print(f'Starting training scheme: {savename}')

    # Set up inputs
    if trainargs is None:
        trainargs = {}
    if pretrainargs is None:
        pretrainargs = {}

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

    # Limit dataset size for testing
    if max_samples is not None:
        dataset = Subset(dataset, indices=torch.randperm(len(dataset))[:max_samples])
        
    # Get 5-fold cross-validation split from ffcvID
    splits = random_split(dataset, [len(dataset) // 5] * 4 + [len(dataset) - 4 * (len(dataset) // 5)])
    testID = ffcvID
    valID = (ffcvID + 1) % 5
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
            **modelargs
        )
        pretrainargs['dae'] = True  # Denoising autoencoder
    elif modelID == 'unet':
        # UNet
        model = UNet(
            in_channels=in_channels, 
            out_channels=out_channels,
            **modelargs
        )
        pretrainargs['dae'] = True  # Denoising autoencoder
    elif modelID == 'resnet':
        # ResNet
        model = ResNet(
            in_channels=in_channels, 
            out_channels=out_channels,
            **modelargs
        )
        pretrainargs['dae'] = True
    elif modelID == 'vit':
        model = VisionTransformer(
            img_size=img_size, 
            in_channels=in_channels, 
            out_channels=out_channels,
            **modelargs
        )
        pretrainargs['mae'] = True  # Masked autoencoder
    model = model.to(device)

    # Pretrain as autoencoder if necessary
    if pretrain:
        if verbose:
            print('Pretraining model as autoencoder.')

        # Modify output layer
        model.set_output_layer(in_channels)

        # Pretrain model
        model, statistics = train_model(
            model, datasets, os.path.join(outpath, f'{savename}_pretrain.pth'),
            segmentation=False, autoencoder=True,
            verbose=verbose, plot=plot,
            n_epochs=n_epochs,
            **pretrainargs
        )

        # Save statistics as json
        if verbose:
            print('Saving pretraining statistics.')
        with open(os.path.join(outpath, f'{savename}_pretrain.json'), 'w') as f:
            json.dump(statistics, f, default=convert_to_serializable)

        # Modify output layer back
        model.set_output_layer(out_channels)
        

    # Train model
    model, statistics = train_model(
        model, datasets, os.path.join(outpath, f'{savename}.pth'),
        segmentation=True,
        verbose=verbose, plot=plot,
        n_epochs=n_epochs,
        **trainargs
    )

    # Test model
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
    
    # Get job id from sys
    jobID = 22
    if len(sys.argv) > 1:
        jobID = int(sys.argv[1])
        
    # Set up datasets, models, and options
    datasets = ['bdello', 'retinas', 'neurons']
    model_options = [
        # ConvolutionalNet
        ['conv', {'n_layers': 8, 'activation': 'relu'}],
        ['conv', {'n_layers': 16, 'activation': 'relu'}],
        ['conv', {'n_layers': 32, 'activation': 'relu'}],
        ['conv', {'n_layers': 8, 'activation': 'gelu'}],
        ['conv', {'n_layers': 16, 'activation': 'gelu'}],
        ['conv', {'n_layers': 32, 'activation': 'gelu'}],
        # VisionTransformer
        ['vit', {'img_size': 128, 'n_layers': 8, 'n_features': 64}],
        ['vit', {'img_size': 128, 'n_layers': 16, 'n_features': 64}],
        ['vit', {'img_size': 128, 'n_layers': 32, 'n_features': 64}],
        ['vit', {'img_size': 128, 'n_layers': 16, 'n_features': 32}],
    #   ['vit', {'img_size': 128, 'n_layers': 16, 'n_features': 64}],
        ['vit', {'img_size': 128, 'n_layers': 16, 'n_features': 128}],
        ['vit', {'img_size': 128, 'n_layers': 16, 'n_features': 64, 'use_cls_token': False}],
        # UNet
        ['unet', {'n_blocks': 2, 'n_layers_per_block': 4, 'expansion': 2}],
        ['unet', {'n_blocks': 3, 'n_layers_per_block': 4, 'expansion': 2}],
        ['unet', {'n_blocks': 4, 'n_layers_per_block': 4, 'expansion': 2}],
        ['unet', {'n_blocks': 2, 'n_layers_per_block': 4, 'expansion': 1}],
        ['unet', {'n_blocks': 3, 'n_layers_per_block': 4, 'expansion': 1}],
        ['unet', {'n_blocks': 4, 'n_layers_per_block': 4, 'expansion': 1}],
        # ResNet
        ['resnet', {'n_blocks': 2, 'n_layers_per_block': 4, 'expansion': 1, 'bottleneck': False}],
        ['resnet', {'n_blocks': 2, 'n_layers_per_block': 8, 'expansion': 1, 'bottleneck': False}],
        ['resnet', {'n_blocks': 2, 'n_layers_per_block': 16, 'expansion': 1, 'bottleneck': False}],
        ['resnet', {'n_blocks': 2, 'n_layers_per_block': 8, 'expansion': 2, 'bottleneck': True}],
        ['resnet', {'n_blocks': 4, 'n_layers_per_block': 4, 'expansion': 2, 'bottleneck': True}],
        ['resnet', {'n_blocks': 6, 'n_layers_per_block': 2, 'expansion': 2, 'bottleneck': True}],
    ]

    # Set up all jobs
    all_jobs = []
    for ffcvid in range(5):
        for dataID in datasets:
            for pretrain in [True, False]:
                for modelID, options in model_options:
                    all_jobs.append((
                        modelID, 
                        dataID, 
                        {**options, 'pretrain':pretrain}, ffcvid
                    ))
    n_jobs = len(all_jobs)

    # Get job parameters
    modelID, dataID, options, ffcvid = all_jobs[jobID]

    # Configure savename
    savename = f'{modelID}_{dataID}'
    for key in sorted(list(options.keys())):
        savename += f'_{key}={options[key]}'
    savename += f'_ffcv={ffcvid}'

    # Run training scheme
    run_training_scheme(
        modelID, 
        dataID, 
        savename,
        n_epochs=100,
        verbose=True,
        **options
    )

    # Done
    print('Done.')

