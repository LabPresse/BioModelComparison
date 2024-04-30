
# Import libraries
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import local modules
from helper_functions import plot_images, plot_roc_curve, get_savename
from datasets.bdello import BdelloDataset
from datasets.retinas import RetinaDataset
from datasets.neurons import NeuronsDataset
from models.conv import ConvolutionalNet
from models.unet import UNet
from models.resnet import ResNet
from models.vit import VisionTransformer

# Set environment
root = 'outfiles/'


# Plot ROC curves
def plot_roc_curves(dataset, model_options):

    # Set up plot
    fig, ax = plt.subplots(1, 1, squeeze=False)
    plt.ion()
    plt.show()

    # Make list of colors the same length as model_options
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_options)))

    # Loop over models and options
    for i, (model, options) in enumerate(model_options):
        # Loop over ffcv splits
        for ffcvid in range(5):

            # Load statistics
            savename = get_savename(model, dataset, options, ffcvid)
            with open(os.path.join(root, f'{savename}.json'), 'r') as f:
                statistics = json.load(f)

            # Get metrics
            fpr = statistics['test_metrics']['roc_fpr']
            tpr = statistics['test_metrics']['roc_tpr']
            auc = statistics['test_metrics']['auc_score']
            accuracy = statistics['test_metrics']['accuracy']
            specificity = statistics['test_metrics']['specificity']
            sensitivity = statistics['test_metrics']['sensitivity']

            # Set label
            label = (
                ('_' if ffcvid != 0 else '')
                + f'{model}; AUC={auc:.3e}; Acc={accuracy:.3e}\nSens={sensitivity:.3e}; Spec={specificity:.3e}'
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
def plot_losses(dataset, model_options):

    # Set up plot
    fig, ax = plt.subplots(2, 1, squeeze=False, sharex=True)
    plt.ion()
    plt.show()

    # Make list of colors the same length as model_options
    colors = plt.cm.viridis(np.linspace(0, 1, len(model_options)))

    # Loop over models and options
    for i, (model, options) in enumerate(model_options):
        # Loop over ffcv splits
        for ffcvid in range(5):

            # Load statistics
            savename = get_savename(model, dataset, options, ffcvid)
            with open(os.path.join(root, f'{savename}.json'), 'r') as f:
                statistics = json.load(f)

            # Plot training and validation losses
            label = ('_' if ffcvid != 0 else '') + f'{model}'
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
def plot_outputs(datasetID, model_options, n_images=5):

    # Get dataset
    if datasetID == 'bdello':
        dataset = BdelloDataset(crop=(256, 256), scale=2)
    elif datasetID == 'retinas':
        dataset = RetinaDataset(crop=(256, 256), scale=2)
    elif datasetID == 'neurons':
        dataset = NeuronsDataset(crop=(256, 256), scale=2)
    else:
        raise ValueError('Invalid dataset.')

    # Get batch
    dataloader = DataLoader(dataset, batch_size=n_images, shuffle=True)
    x, y = next(iter(dataloader))

    # Loop over models and options
    zs = {}
    for i, (modelID, options) in enumerate(model_options):
        
        # Load model
        ops = options.copy()
        if 'pretrain' in ops:
            del ops['pretrain']
        if modelID == 'conv':
            model = ConvolutionalNet(in_channels=3, out_channels=2, **ops)
        elif modelID == 'unet':
            model = UNet(in_channels=3, out_channels=2, **ops)
        elif modelID == 'resnet':
            model = ResNet(in_channels=3, out_channels=2, **ops)
        elif modelID == 'vit':
            model = VisionTransformer(in_channels=3, out_channels=2, **ops)

        # Load model
        savename = get_savename(modelID, datasetID, options, 0)
        model.load_state_dict(torch.load(os.path.join(root, f'{savename}.pth')))
        model.eval()
        
        # Get predictions
        if modelID == 'vit':
            # If vit, split image into 4 quarters, predict then merge
            z0 = model(x[:, :, :128, :128]).detach().cpu().numpy().argmax(axis=1)
            z1 = model(x[:, :, 128:, :128]).detach().cpu().numpy().argmax(axis=1)
            z2 = model(x[:, :, :128, 128:]).detach().cpu().numpy().argmax(axis=1)
            z3 = model(x[:, :, 128:, 128:]).detach().cpu().numpy().argmax(axis=1)
            z01 = np.concatenate((z0, z1), axis=1)
            z23 = np.concatenate((z2, z3), axis=1)
            z = np.concatenate((z01, z23), axis=2)
        else:
            z = model(x).detach().cpu().numpy().argmax(axis=1)
        zs[modelID] = z

    # Plot images
    fig, ax = plot_images(Images=x, Targets=y, **zs)

    # Return figure and axes
    return fig, ax

# Main
if __name__ == '__main__':
    
    # Set up datasets, models, and options
    model_options = [
        # ConvolutionalNet
        # ['conv', {'pretrain': False, 'n_layers': 8, 'activation': 'relu'}],
        ['conv', {'pretrain': False, 'n_layers': 16, 'activation': 'relu'}],
        # ['conv', {'pretrain': False, 'n_layers': 32, 'activation': 'relu'}],
        # ['conv', {'pretrain': False, 'n_layers': 8, 'activation': 'gelu'}],
        # ['conv', {'pretrain': False, 'n_layers': 16, 'activation': 'gelu'}],
        # ['conv', {'pretrain': False, 'n_layers': 32, 'activation': 'gelu'}],
        # # VisionTransformer
        ['vit', {'pretrain': False, 'img_size': 128, 'n_layers': 8, 'n_features': 64}],
        # ['vit', {'pretrain': False, 'img_size': 128, 'n_layers': 16, 'n_features': 64}],
        # ['vit', {'pretrain': False, 'img_size': 128, 'n_layers': 32, 'n_features': 64}],
        # ['vit', {'pretrain': False, 'img_size': 128, 'n_layers': 16, 'n_features': 32}],
        # ['vit', {'pretrain': False, 'img_size': 128, 'n_layers': 16, 'n_features': 128}],
        # ['vit', {'pretrain': False, 'img_size': 128, 'n_layers': 16, 'n_features': 64, 'use_cls_token': False}],
        # # UNet
        # ['unet', {'pretrain': False'n_blocks': 2, 'n_layers_per_block': 4}],
        ['unet', {'pretrain': False, 'n_blocks': 3, 'n_layers_per_block': 4}],
        # ['unet', {'pretrain': False, 'n_blocks': 4, 'n_layers_per_block': 4}],
        # ['unet', {'pretrain': False, 'n_blocks': 2, 'n_layers_per_block': 4, 'expansion': 1}],
        # ['unet', {'pretrain': False, 'n_blocks': 3, 'n_layers_per_block': 4, 'expansion': 1}],
        # ['unet', {'pretrain': False, 'n_blocks': 4, 'n_layers_per_block': 4, 'expansion': 1}],
        # # ResNet
        ['resnet', {'pretrain': False, 'n_blocks': 2, 'n_layers_per_block': 4, 'expansion': 1, 'bottleneck': False}],
        # ['resnet', {'pretrain': False, 'n_blocks': 2, 'n_layers_per_block': 8, 'expansion': 1, 'bottleneck': False}],
        # ['resnet', {'pretrain': False, 'n_blocks': 2, 'n_layers_per_block': 16, 'expansion': 1, 'bottleneck': False}],
        # ['resnet', {'pretrain': False, 'n_blocks': 2, 'n_layers_per_block': 8, 'expansion': 2, 'bottleneck': True}],
        # ['resnet', {'pretrain': False, 'n_blocks': 4, 'n_layers_per_block': 4, 'expansion': 2, 'bottleneck': True}],
        # ['resnet', {'pretrain': False, 'n_blocks': 6, 'n_layers_per_block': 2, 'expansion': 2, 'bottleneck': True}],
    ]
    
    # Plot loss
    dataset = 'neurons'
    # fig0, ax0 = plot_losses(dataset, model_options)
    fig1, ax1 = plot_roc_curves(dataset, model_options)
    # fig2, ax2 = plot_outputs(dataset, model_options)

    # Done
    print('Done.')
