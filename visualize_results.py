
# Import libraries
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import local modules
from helper_functions import plot_images, plot_roc_curve

# Set environment
root = 'outfiles/'


# Main
if __name__ == '__main__':
    
    # Select results
    savename = 'conv_retinas_pretrain=True'

    # Load model
    model = torch.load(os.path.join(root, f'{savename}.pth'))

    # Load statistics
    with open(os.path.join(root, f'{savename}.json'), 'r') as f:
        statistics = json.load(f)

    # Plot training and validation losses
    fig, ax = plt.subplots(1, 1)
    plt.ion()
    plt.show()
    ax.plot(statistics['train_losses'], label='Train')
    ax.plot(statistics['val_losses'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()

    # Plot ROC curve
    accuracy = statistics['test_metrics']['accuracy']
    sensitivity = statistics['test_metrics']['sensitivity']
    specificity = statistics['test_metrics']['specificity']
    auc_score = statistics['test_metrics']['auc_score']
    fpr = statistics['test_metrics']['roc_fpr']
    tpr = statistics['test_metrics']['roc_tpr']
    fig2 = plt.figure()
    plot_roc_curve(fpr, tpr, accuracy, sensitivity, specificity, auc_score)

    # Done
    print('Done.')
