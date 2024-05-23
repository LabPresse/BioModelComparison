
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
from models.conv import ConvolutionalNet
from models.unet import UNet
from models.resnet import ResNet
from models.vit import VisionTransformer

# Set environment
root = 'cluster/outfiles/'
figpath = 'figures/'



# Plot ROC curves
def plot_roc_curves(basenames):

    # Set up plot
    fig, ax = plt.subplots(1, 1, squeeze=False)
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
        for ffcvid in range(1):
            savename = basename + f'_ffcv={ffcvid}'
            with open(os.path.join(root, f'{savename}.json'), 'r') as f:
                statistics = json.load(f)
            auc_avg += statistics['test_metrics']['auc_score'] / 5
            acc_avg += statistics['test_metrics']['accuracy'] / 5
            sens_avg += statistics['test_metrics']['sensitivity'] / 5
            spec_avg += statistics['test_metrics']['specificity'] / 5


        # Loop over ffcv splits
        for ffcvid in range(1):

            # Load statistics
            savename = basename + f'_ffcv={ffcvid}'
            with open(os.path.join(root, f'{savename}.json'), 'r') as f:
                statistics = json.load(f)

            # Get metrics
            fpr = statistics['test_metrics']['roc_fpr']
            tpr = statistics['test_metrics']['roc_tpr']

            # Set label
            label = (
                ('_' if ffcvid != 0 else '')
                + f'{basename}\n'
                + '; '.join([
                    f'AUC={auc_avg:.3f}',
                    f'Acc={acc_avg:.3f}',
                    f'Sens={sens_avg:.3f}',
                    f'Spec={spec_avg:.5f}'
                ])
            )

            # Plot ROC curve
            ax[0, 0].plot(fpr, tpr, label=label, color=colors[i])

    # Set labels
    ax[0, 0].set_ylabel('TPR')
    ax[0, 0].set_xlabel('FPR')
    ax[0, 0].legend()
    plt.pause(.1)
    plt.tight_layout()

    # Return figure and axes
    return fig, ax


# Plot loss
def plot_losses(basenames):

    # Set up plot
    fig, ax = plt.subplots(2, 1, squeeze=False, sharex=True)
    plt.ion()
    plt.show()

    # Make list of colors the same length as model_options
    colors = plt.cm.viridis(np.linspace(0, 1, len(basenames)))

    # Loop over models and options
    for i, basename in enumerate(basenames):
        # Loop over ffcv splits
        for ffcvid in range(1):

            # Load statistics
            savename = basename + f'_ffcv={ffcvid}'
            with open(os.path.join(root, f'{savename}.json'), 'r') as f:
                statistics = json.load(f)

            # Plot training and validation losses
            label = ('_' if ffcvid != 0 else '') + f'{basename}'
            ax[0, 0].plot(statistics['train_losses'], label=label, color=colors[i])
            ax[1, 0].plot(statistics['val_losses'], label=label, color=colors[i])

    # Set labels
    ax[0, 0].set_ylabel('Train Loss')
    ax[1, 0].set_ylabel('Val Loss')
    ax[1, 0].set_xlabel('Epoch')
    ax[0, 0].legend()
    plt.pause(.1)
    plt.tight_layout()

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
        
        # Load model
        if 'conv' in basename:
            n_layers = int(basename.replace('n_layers=', 'xxx').split('xxx')[1].split('_')[0])
            model = ConvolutionalNet(in_channels=in_channels, out_channels=out_channels, n_layers=n_layers)
        elif 'unet' in basename:
            n_blocks = int(basename.replace('n_blocks=', 'xxx').split('xxx')[1].split('_')[0])
            model = UNet(in_channels=in_channels, out_channels=out_channels, n_blocks=n_blocks)
        elif 'resnet' in basename:
            n_blocks = int(basename.replace('n_blocks=', 'xxx').split('xxx')[1].split('_')[0])
            model = ResNet(in_channels=in_channels, out_channels=out_channels, n_blocks=n_blocks)
        elif 'vit' in basename or 'vim' in basename:
            n_layers = int(basename.replace('n_layers=', 'xxx').split('xxx')[1].split('_')[0])
            model = VisionTransformer(img_size=128, in_channels=in_channels, out_channels=out_channels, n_layers=n_layers)

        # Load model
        savename = f'{basename}_ffcv=0'
        model.load_state_dict(torch.load(os.path.join(root, f'{savename}.pth')))
        model.eval()
        
        # Get predictions
        z = model(x).detach().cpu().numpy().argmax(axis=1)
        zs[modelID] = z

    # Plot images
    fig, ax = plot_images(Images=x, Targets=y, **zs)

    # Return figure and axes
    return fig, ax


# Run training scheme
if __name__ == "__main__":

    # Set up constants
    datasets = ['bdello', 'neurons', ]#'retinas']
    models = ['conv', 'unet', 'resnet', 'vit']

    # Set up basenames for all jobs
    basenames = [f[:-5] for f in os.listdir(root) if f.endswith('.json')]
    basenames = [f[:-7] for f in basenames if f.endswith('ffcv=0')]

    # Set up best models
    best_params = {
        'conv': 'n_layers=8',
        'unet': 'n_blocks=3',
        'resnet': 'n_blocks=3',
        'vit': 'n_layers=8',
    }
    best_basenames = []
    for modelID in models:
        best_basenames += [f for f in basenames if modelID in f and best_params[modelID] in f]


    # Loop over datasets
    for datasetID in datasets:

        # Get basenames for dataset
        basenames_dataset = [f for f in basenames if datasetID in f]
        best_basenames_dataset = [f for f in best_basenames if datasetID in f]

        # Plot ROC curves and losses for different params of each model
        for modelID in models:

            # Get models for dataset
            basenames_dataset_model = sorted([f for f in basenames_dataset if modelID in f])

            # Plot ROC curves
            fig, ax = plot_roc_curves(basenames_dataset_model)
            fig.savefig(os.path.join(figpath, f'{datasetID}_{modelID}_roc.png'))

            # Plot losses
            fig, ax = plot_losses(basenames_dataset_model)
            fig.savefig(os.path.join(figpath, f'{datasetID}_{modelID}_losses.png'))

        # Plot ROC curves for best models
        fig, ax = plot_roc_curves(best_basenames_dataset)
        fig.savefig(os.path.join(figpath, f'{datasetID}_best_roc.png'))

        # Plot losses for best models
        fig, ax = plot_losses(best_basenames_dataset)
        fig.savefig(os.path.join(figpath, f'{datasetID}_best_losses.png'))

        # Plot outputs for best models
        fig, ax = plot_outputs(datasetID, best_basenames_dataset)
        fig.savefig(os.path.join(figpath, f'{datasetID}_best_outputs.png'))

    # Done
    print('Done.')
