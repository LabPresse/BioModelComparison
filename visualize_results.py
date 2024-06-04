
# Import libraries
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import local modules
from helper_functions import plot_images, plot_roc_curve, get_savename
from data.bdello import BdelloDataset
from data.retinas import RetinaDataset
from data.neurons import NeuronsDataset
from data.letters import ChineseCharacters
from models.conv import ConvolutionalNet
from models.unet import UNet
from models.resnet import ResNet
from models.vit import VisionTransformer
from models.vim import VisionMamba

# Set environment
root = 'cluster/outfiles/'
figpath = 'figures/'


# Plot training stats
def plot_training_stats(basenames, ffcvIDs=None):
    """Plot the losses and roc curves in a single figure."""

    # Set up ffcvIDs
    if ffcvIDs is None:
        ffcvIDs = [i for i in range(5)]

    # Set up figure
    fig, ax = plt.subplots(1, 3, squeeze=False)
    fig.set_size_inches(15, 5)
    plt.ion()
    plt.show()

    # Make list of colors the same length as model_options
    colors = plt.cm.viridis(np.linspace(0, 1, len(basenames)))

    # Loop over models and options
    for i, basename in enumerate(basenames):

        # Get avg stats
        auc_avg = 0
        acc_avg = 0
        sens_avg = 0
        spec_avg = 0
        for ffcvid in ffcvIDs:
            savename = basename + f'_ffcv={ffcvid}'
            with open(os.path.join(root, f'{savename}.json'), 'r') as f:
                statistics = json.load(f)
            auc_avg += statistics['test_metrics']['auc_score'] / len(ffcvIDs)
            acc_avg += statistics['test_metrics']['accuracy'] / len(ffcvIDs)
            sens_avg += statistics['test_metrics']['sensitivity'] / len(ffcvIDs)
            spec_avg += statistics['test_metrics']['specificity'] / len(ffcvIDs)

        # Loop over ffcv splits
        for ffcvid in ffcvIDs:

            # Load statistics
            savename = basename + f'_ffcv={ffcvid}'
            with open(os.path.join(root, f'{savename}.json'), 'r') as f:
                statistics = json.load(f)

            ### ROC CURVE ###

            # Get metrics
            fpr = statistics['test_metrics']['roc_fpr']
            tpr = statistics['test_metrics']['roc_tpr']

            # Set label
            label = (
                ('_' if ffcvid != 0 else '')
                + f'{basename}\n'
                + '; '.join([
                    f'AUC={auc_avg:.2f}',
                    f'Acc={acc_avg:.2f}',
                    f'Sens={sens_avg:.2f}',
                    f'Spec={spec_avg:.2f}'
                ])
            )

            # Plot ROC curve
            ax[0, 0].plot(fpr, tpr, label=label, color=colors[i])

            ### LOSSES ###

            # Plot training and validation losses
            label = ('_' if ffcvid != 0 else '') + f'{basename}'
            ax[0, 1].semilogy(statistics['train_losses'], label=label, color=colors[i])
            ax[0, 2].semilogy(statistics['val_losses'], label=label, color=colors[i])


    # Finalize ROC plot
    ax[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
    ax[0, 0].set_title('ROC Curve')
    ax[0, 0].set_ylabel('TPR')
    ax[0, 0].set_xlabel('FPR')
    ax[0, 0].legend()

    # Finalize loss plots
    ax[0, 1].set_title('Training Loss')
    ax[0, 1].set_ylabel('Log Loss')
    ax[0, 1].set_xlabel('Epoch')
    ax[0, 2].set_title('Validation Loss')
    ax[0, 2].set_ylabel('Log Loss')
    ax[0, 2].set_xlabel('Epoch')
    ax[0, 1].legend()
    ax[0, 2].legend()

    # Finalize figure
    plt.tight_layout()
    plt.pause(1)

    # Return figure and axes
    return fig, ax


# Plot outputs
def plot_outputs(datasetID, basenames, n_images=5):

    # Get dataset
    if datasetID == 'bdello':
        dataset = BdelloDataset(crop=(128, 128), scale=4)
    elif datasetID == 'retinas':
        dataset = RetinaDataset(crop=(128, 128), scale=4)
    elif datasetID == 'neurons':
        dataset = NeuronsDataset(crop=(128, 128), scale=2)
    elif datasetID == 'letters':
        dataset = ChineseCharacters(shape=(128, 128), sigma=0.25, max_characters=1000)

    # Get batch
    dataloader = DataLoader(dataset, batch_size=n_images, shuffle=True)
    x, y = next(iter(dataloader))
    in_channels = x.shape[1]
    out_channels = 2

    # Loop over models and options
    zs = {}
    for i, basename in enumerate(basenames):

        # Get modelID
        modelID = basename.split('_')[0]

        # Extract model options
        options = {}
        if 'n_layers' in basename:
            options['n_layers'] = int(basename.replace('n_layers=', 'xxx').split('xxx')[1].split('_')[0])
        if 'n_blocks' in basename:
            options['n_blocks'] = int(basename.replace('n_blocks=', 'xxx').split('xxx')[1].split('_')[0])
        if 'n_features' in basename:
            options['n_features'] = int(basename.replace('n_features=', 'xxx').split('xxx')[1].split('_')[0])
        if 'expansion' in basename:
            options['expansion'] = int(basename.replace('expansion=', 'xxx').split('xxx')[1].split('_')[0])

        # Load model
        if 'conv' in basename:
            model = ConvolutionalNet(
                in_channels=in_channels, out_channels=out_channels, **options
            )
        elif 'unet' in basename:
            model = UNet(
                in_channels=in_channels, out_channels=out_channels, **options
            )
        elif 'resnet' in basename:
            model = ResNet(
                in_channels=in_channels, out_channels=out_channels, **options
            )
        elif 'vit' in basename:
            model = VisionTransformer(
                img_size=128, in_channels=in_channels, out_channels=out_channels, **options
            )
        elif 'vim' in basename:
            model = VisionMamba(
                img_size=128, in_channels=in_channels, out_channels=out_channels, **options
            )

        # Load model
        savename = f'{basename}_ffcv=0'
        model.load_state_dict(torch.load(os.path.join(root, f'{savename}.pth')), strict=False)
        model.eval()
        
        # Get predictions
        z = model(x).detach().cpu().numpy().argmax(axis=1)
        zs[modelID] = z

    # Plot images
    fig = plt.figure()
    fig, ax = plot_images(Images=x, Targets=y, **zs)

    # Return figure and axes
    return fig, ax


# Run training scheme
if __name__ == "__main__":

    # Set up constants
    datasets = ['letters', 'bdello', 'neurons', 'retinas']
    models = ['conv', 'unet', 'vit', 'vim',]

    # Set up basenames for all jobs
    basenames = [f[:-5] for f in os.listdir(root) if f.endswith('.json')]
    basenames = [f[:-7] for f in basenames if f.endswith('ffcv=0')]

    # Set up best models
    best_params = {
        'conv': 'n_layers=8',
        'unet': 'n_blocks=3',
        'vit': 'n_layers=6',
        'vim': 'n_layers=6',
    }

    # Set up ffvIDs
    ffcvIDs = [0]

    # Loop over datasets
    for datasetID in datasets:

        # Get basenames for dataset
        basenames_dataset = [f for f in basenames if datasetID in f]
        best_basenames_dataset = [f'{modelID}_{datasetID}_{best_params[modelID]}' for modelID in models]

        # Plot training stats for best models
        fig, ax = plot_training_stats(best_basenames_dataset, ffcvIDs=ffcvIDs)
        fig.savefig(os.path.join(figpath, f'{datasetID}_best_training_stats.png'))

        # Plot outputs for best models
        fig, ax = plot_outputs(datasetID, best_basenames_dataset)
        fig.savefig(os.path.join(figpath, f'{datasetID}_best_outputs.png'))

        # Plot ROC curves and losses for different params of each model
        for modelID in models:

            # Get models for dataset
            basenames_dataset_model = sorted([f for f in basenames_dataset if modelID in f])

            # Plot training stats
            fig, ax = plot_training_stats(basenames_dataset_model, ffcvIDs=ffcvIDs)
            fig.savefig(os.path.join(figpath, f'{datasetID}_{modelID}_training_stats.png'))

    # Done
    print('Done.')
