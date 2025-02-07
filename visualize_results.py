
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
from data.letters import LettersDataset
from models.conv import ConvolutionalNet
from models.unet import UNet
from models.resnet import ResNet
from models.vit import VisionTransformer
from models.vim import VisionMamba

# Set environment
root = 'cluster/outfiles/'
figpath = 'figures/'

# Set random seed for torch and numpy
seed = 1993
torch.manual_seed(seed)
np.random.seed(seed)


# Plot training stats
def plot_training_stats(basenames_dict, ffcvIDs=None, model_labels=None):
    """Plot the losses and roc curves in a single figure."""

    # Set up ffcvIDs
    if ffcvIDs is None:
        ffcvIDs = [i for i in range(5)]

    # If basenames_dict is list, convert to dict
    if isinstance(basenames_dict, list):
        basenames_dict = {None: basenames_dict}
    n_rows = len(basenames_dict.keys())

    # Set up model_labels
    if model_labels is None:
        model_labels = {basename: basename for basename in basenames_dict.values()}

    # Set up figure
    fig, ax = plt.subplots(n_rows, 3, squeeze=False)
    fig.set_size_inches(15, 4*n_rows)
    plt.ion()
    plt.show()

    # Loop over keys
    for keyid, (key, basenames_i) in enumerate(basenames_dict.items()):

        # Make list of colors the same length as model_options
        colors = plt.cm.viridis(np.linspace(0, 1, len(basenames_i)))

        # Loop over models and options
        for i, basename in enumerate(basenames_i):

            # Get avg stats
            n_avg = 0
            auc_avg = 0
            acc_avg = 0
            sens_avg = 0
            spec_avg = 0
            loss_train_avg = 0
            loss_val_avg = 0
            train_time_avg = 0
            fpr_base = np.linspace(0, 1, 500)
            tpr_avg = np.zeros(500)
            for ffcvid in ffcvIDs:
                savename = basename + f'_ffcv={ffcvid}'
                try:
                    with open(os.path.join(root, f'{savename}.json'), 'r') as f:
                        statistics = json.load(f)
                except:
                    print(f'Could not load {savename}.json')
                    continue
                n_avg += 1
                auc_avg += statistics['test_metrics']['auc_score']
                acc_avg += statistics['test_metrics']['accuracy']
                sens_avg += statistics['test_metrics']['sensitivity']
                spec_avg += statistics['test_metrics']['specificity']
                loss_train_avg += np.array(statistics['train_losses'])
                loss_val_avg += np.array(statistics['val_losses'])
                train_time_avg += sum(statistics['epoch_times'])
                tpr_avg += np.interp(
                    fpr_base, 
                    statistics['test_metrics']['roc_fpr'], 
                    statistics['test_metrics']['roc_tpr']
                )
                n_parameters = statistics['n_parameters']
            auc_avg /= n_avg
            acc_avg /= n_avg
            sens_avg /= n_avg
            spec_avg /= n_avg
            loss_train_avg /= n_avg
            loss_val_avg /= n_avg
            train_time_avg /= n_avg
            tpr_avg /= n_avg
            tpr_avg[0] = 0



            # Make label
            label = (
                f'{model_labels[basename]} ({n_parameters} parameters)\n'
                + '\n'.join([
                    f'Train Time={int(train_time_avg/60)} mins',
                    f'AUC={auc_avg:.3f}; SENS={sens_avg:.3f}',
                    f'ACC={acc_avg:.3f}; SPEC={spec_avg:.4f}'
                ])
            )

            # Plot curves
            ax[keyid, 0].semilogy(loss_train_avg, color=colors[i])
            ax[keyid, 1].semilogy(loss_val_avg, color=colors[i])
            ax[keyid, 2].plot(fpr_base, tpr_avg, label=label, color=colors[i])

        # Get prefix
        prefix = '' if key is None else f'{key.upper()}\n'

        # Finalize loss plots
        if keyid == 0:
            ax[keyid, 0].set_title('Training Loss')
            ax[keyid, 1].set_title('Validation Loss')
            ax[keyid, 2].set_title('ROC Curve')
            ax[keyid, 0].set_xlabel('Epoch')
            ax[keyid, 1].set_xlabel('Epoch')
            ax[keyid, 2].set_xlabel('False Positive Rate')
        ax[keyid, 0].set_ylabel(prefix+'Log Loss')
        ax[keyid, 1].set_ylabel('Log Loss')
        ax[keyid, 2].set_ylabel('True Positive Rate')
        ax[keyid, 2].plot([0, 1], [0, 1], 'k--', label='Random')
        ax[keyid, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Finalize figure
    plt.tight_layout()
    plt.pause(1)

    # Return figure and axes
    return fig, ax


# Plot outputs
def plot_outputs(datasetID, basenames, n_images=3, model_labels=None):

    # Set up model_labels
    if model_labels is None:
        model_labels = {basename: basename.split('_')[0] for basename in basenames}

    # Get dataset
    if datasetID == 'bdello':
        dataset = BdelloDataset(crop=(128, 128), scale=4)
    elif datasetID == 'retinas':
        dataset = RetinaDataset(crop=(512, 512), scale=4)
    elif datasetID == 'neurons':
        dataset = NeuronsDataset(crop=(512, 512), scale=2)
    elif datasetID == 'letters':
        dataset = LettersDataset(shape=(128, 128), sigma=0.25)

    # Get batch
    dataloader = DataLoader(dataset, batch_size=n_images, shuffle=True)
    tries = 0
    while tries < 10:
        x, y = next(iter(dataloader))
        if (y.sum(axis=(1, 2)) > 0).sum(axis=0) >=3:
            break
    in_channels = x.shape[1]
    out_channels = 2

    # Loop over models and options
    zs = {}
    for i, basename in enumerate(basenames):

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
        model.load_state_dict(
            torch.load(
                os.path.join(root, f'{savename}.pth'),
                map_location=torch.device('cpu'),
            ), 
            strict=False
        )
        model.eval()

        # Split x into 128x128 patches and get predictions
        z = np.zeros((x.shape[0], x.shape[2], x.shape[3]))
        for i in range(x.shape[2]//128):
            for j in range(x.shape[3]//128):
                x_patch = x[:, :, i*128:(i+1)*128, j*128:(j+1)*128]
                z_patch = model(x_patch).detach().cpu().numpy().argmax(axis=1)
                z[:, i*128:(i+1)*128, j*128:(j+1)*128] = z_patch

        # Add to zs
        zs[model_labels[basename]] = z

    # If dataset is neurons, increase the brightness of the images
    if datasetID == 'neurons':
        x = np.asarray(x)
        x -= x.min(axis=(2, 3), keepdims=True)
        x /= x.max(axis=(2, 3), keepdims=True)
        x = np.clip(5*x, 0, 1)

    # Plot images
    fig = plt.figure()
    plt.ion()
    plt.show()
    fig, ax = plot_images(Images=x, Targets=y, **zs, transpose=True)

    # Return figure and axes
    return fig, ax


# Generate plots
if __name__ == "__main__":

    ### Plot robustness ###

    # Initialize lists
    blurs = [4, 8, 16]
    sigmas = [0.5, 1, 2]
    models = ['conv', 'unet', 'vit', 'vim']

    # Loop over models
    for modelID in models:

        # Set up figure for examples
        fig, ax = plt.subplots(len(blurs), len(sigmas), squeeze=False, figsize=(9, 9))
        plt.ion()
        plt.show()

        # Set up figures for stats
        fig_stats, ax_stats = plt.subplots(len(blurs), len(sigmas), squeeze=False, figsize=(9, 9))
        plt.ion()
        plt.show()

        # Loop over datasets
        for i, blur in enumerate(blurs):
            for j, sigma in enumerate(sigmas):

                # Get datasetID
                datasetID = f'letters_blur={blur}_sigma={sigma}'

                # Load dataset
                dataset = LettersDataset(shape=(128, 128), sigma=sigma, blur=blur)
                        
                # Get basenames
                basenames = [f for f in os.listdir(root) if modelID in f]
                basenames = [f for f in basenames if datasetID in f]
                basenames = [f for f in basenames if f.endswith('ffcv=0.json')]

                # Get best model
                best_loss = None
                best_model = None
                for basename in basenames:
                    with open(os.path.join(root, basename), 'r') as f:
                        statistics = json.load(f)
                    if (best_loss is None) or (statistics['min_val_loss'] < best_loss):
                        best_loss = statistics['min_val_loss']
                        best_model = '_'.join(basename.split('_')[:-1])

                # Get test metrics
                acc_avg = 0
                sens_avg = 0
                spec_avg = 0
                auc_avg = 0
                for ffid in range(3):
                    with open(os.path.join(root, f'{best_model}_ffcv={ffid}.json'), 'r') as f:
                        statistics = json.load(f)
                    acc_avg += statistics['test_metrics']['accuracy'] / 5
                    sens_avg += statistics['test_metrics']['sensitivity'] / 5
                    spec_avg += statistics['test_metrics']['specificity'] / 5
                    auc_avg += statistics['test_metrics']['auc_score'] / 5

                # Plot bar graph of test metrics
                ax_stats[i, j].bar(['ACC', 'SENS', 'SPEC', 'AUC'], [acc_avg, sens_avg, spec_avg, auc_avg])

                # Reset basename
                basename = best_model

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
                in_channels = 1
                out_channels = 2
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
                model.load_state_dict(
                    torch.load(
                        os.path.join(root, f'{savename}.pth'),
                        map_location=torch.device('cpu'),
                    ), 
                    strict=False
                )
                model.eval()

                # Get batch
                x, y = next(iter(DataLoader(dataset, batch_size=3, shuffle=True)))

                # Get predictions
                z = model(x).detach().cpu().numpy().argmax(axis=1)

                # Plot example
                img = np.zeros((128, 128, 3))
                img[:, :, 1] = x[0, 0].detach().cpu().numpy()  # Green channel for input
                img[:, :, 0] = z[0]  # Red channel for prediction
                # img[:, :, 2] = y[0]  # Blue channel for target
                ax[i, j].imshow(img)
            
        # Finalize figures
        fig.suptitle(f'{modelID.upper()}')
        fig_stats.suptitle(f'{modelID.upper()}')
        for i in range(len(blurs)):
            for j in range(len(sigmas)):
                ax_title = f'Blur={blurs[i]} pixels; SNR={1/sigmas[j]:.2f}'
                ax[i, j].set_title(ax_title)
                ax_stats[i, j].set_title(ax_title)
                ax[i, j].axis('off')
                ax_stats[i, j].set_ylim([0,1])
        fig.tight_layout()
        plt.pause(.1)
        fig_stats.tight_layout()
        plt.pause(.1)
        plt.pause(.1)

        # Save figures
        fig.savefig(os.path.join(figpath, f'robustness_{modelID}_examples.png'), dpi=900)
        fig_stats.savefig(os.path.join(figpath, f'robustness_{modelID}_stats.png'), dpi=900)      

    # Done
    print('Done.')


    ### Plot results ###

    # Set up constants
    datasets = ['bdello', 'letters', 'neurons', 'retinas']
    models = ['conv', 'unet', 'vit', 'vim',]

    # Set up basenames for all jobs
    basenames = [f[:-5] for f in os.listdir(root) if f.endswith('.json')]
    basenames = [f[:-7] for f in basenames if f.endswith('ffcv=0')]

    # Set up model labels
    model_labels_best = {}
    model_labels_all = {}
    for basename in basenames:

        # Extract model features
        options = {}
        modelID = basename.split('_')[0]
        if 'n_layers' in basename:
            options['n_layers'] = int(basename.replace('n_layers=', 'xxx').split('xxx')[1].split('_')[0])
        if 'n_blocks' in basename:
            options['n_blocks'] = int(basename.replace('n_blocks=', 'xxx').split('xxx')[1].split('_')[0])
        if 'n_features' in basename:
            options['n_features'] = int(basename.replace('n_features=', 'xxx').split('xxx')[1].split('_')[0])
        if 'expansion' in basename:
            options['expansion'] = int(basename.replace('expansion=', 'xxx').split('xxx')[1].split('_')[0])

        # Set up model labels
        if modelID == 'conv':
            label_best = 'CNN'
            label_all = f"CNN -- {options['n_layers']} Layers"
        elif modelID == 'unet':
            label_best = 'UNet'
            label_all = f"UNet -- {options['n_blocks']} Blocks"
        elif modelID == 'vit':
            label_best = 'ViT'
            label_all = f"ViT -- {options['n_layers']} Layers"
        elif modelID == 'vim':
            label_best = 'VSSM'
            label_all = f"VSSM -- {options['n_layers']} Layers"

        # Add to model labels
        model_labels_best[basename] = label_best
        model_labels_all[basename] = label_all

    # Loop over datasets
    for datasetID in datasets:

        # Loop over models
        best_basenames_dataset = []
        basenames_dataset_dict = {}
        for modelID in models:

            # Get model basenames
            basenames_dataset = [f for f in basenames if datasetID in f]
            basenames_dataset_model = sorted([f for f in basenames_dataset if modelID in f])
            basenames_dataset_dict[modelID] = basenames_dataset_model
            
            # Get best model
            best_loss = None
            best_model = None
            for basename in basenames_dataset_model:

                # Load statistics
                with open(os.path.join(root, f'{basename}_ffcv=0.json'), 'r') as f:
                    statistics = json.load(f)

                # Update best model
                if (best_loss is None) or (statistics['min_val_loss'] < best_loss):
                    best_loss = statistics['min_val_loss']
                    best_model = basename
            
            # Add best model to list
            best_basenames_dataset.append(best_model)

        # Plot outputs for best models
        fig, ax = plot_outputs(datasetID, best_basenames_dataset, model_labels=model_labels_best)
        fig.savefig(os.path.join(figpath, f'{datasetID}_best_outputs.png'), dpi=900)

        # Plot training stats for best models
        fig, ax = plot_training_stats(best_basenames_dataset, model_labels=model_labels_best)
        fig.savefig(os.path.join(figpath, f'{datasetID}_best_training_stats.png'))

        # Plot all training stats
        fig, ax = plot_training_stats(basenames_dataset_dict, model_labels=model_labels_all)
        fig.savefig(os.path.join(figpath, f'{datasetID}_all_training_stats.png'))

    # Done
    print('Done.')
