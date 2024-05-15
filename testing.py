
# Import libraries
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Import local modules
from helper_functions import plot_roc_curve


# Define a function to evaluate a PyTorch model
def evaluate_model(model, dataset, verbose=True, plot=False):
    """Evaluate a PyTorch model on a dataset."""

    ### EVALUATE THE MODEL ###
    if verbose:
        status = f'Evaluating model with {sum(p.numel() for p in model.parameters())} parameters.'
        print(status)

    # Set up model and device
    model.eval()
    device = next(model.parameters()).device

    # Set up the dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Determine downsample factor
    n_images = len(dataloader)
    n_pixels = math.prod(next(iter(dataloader))[0].shape)
    n_samples = n_images * n_pixels
    if n_samples > 100000:
        n_samples_per_img = 100000 // n_images
    else:
        n_samples_per_img = n_pixels

    # Create lists
    true_labels = []
    predicted_probs = []
    predicted_labels = []

    # Iterate over the dataloader
    for i, (x, y) in enumerate(dataloader):
        if verbose and (((i+1)%100==0) or (i==len(dataloader)-1) or (len(dataloader)<20)):
            print(f'--Image {i + 1}/{len(dataloader)}')

        # Move the data to the device
        x = x.to(device)
        y = y.to(device)

        # Forward pass through the model
        z = model(x)

        # Get the predicted probabilities and labels
        probs = torch.softmax(z, dim=1)
        labels = torch.argmax(probs, dim=1)

        # Flatten the data
        trues = y.cpu().numpy().reshape(-1)
        probs = probs[:, 1].cpu().detach().numpy().reshape(-1)
        labels = labels.cpu().numpy().reshape(-1)

        # Randomly sample the indices
        indices = np.random.choice(len(trues), min(n_samples_per_img, len(trues)), replace=False)
        trues = trues[indices]
        probs = probs[indices]
        labels = labels[indices]

        # Append results to lists
        true_labels.extend(trues)
        predicted_probs.extend(probs)
        predicted_labels.extend(labels)

    # Convert the lists to numpy arrays
    true_labels = np.array(true_labels)
    predicted_probs = np.array(predicted_probs)
    predicted_labels = np.array(predicted_labels)


    ### CALCULATE EVALUATION METRICS ###
    if verbose:
        print('Calculating evaluation metrics.')

    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        true_labels, 
        predicted_labels,
    ).ravel()

    # Calculate sensitivity, specificity, and accuracy
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1score = 2 * tp / (2 * tp + fp + fn)


    ### CALCULATE ROC CURVE AND AUC SCORE ###
    if verbose:
        print('Calculating ROC curve and AUC score.')

    # Calculate the ROC curve and AUC score
    roc_fpr, roc_tpr, roc_thresholds = roc_curve(
        true_labels, 
        predicted_probs,
    )
    auc_score = roc_auc_score(
        true_labels, 
        predicted_probs,
    )

    # Plot the ROC curve
    if plot:
        plot_roc_curve(roc_fpr, roc_tpr, auc_score)


    ### PACKAGE THE METRICS ###
    if verbose:
        print('Packaging the metrics.')

    # Package the metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1score': f1score,
        'auc_score': auc_score,
        'roc_fpr': roc_fpr,
        'roc_tpr': roc_tpr,
        'roc_thresholds': roc_thresholds,
    }

    # Return the evaluation metrics
    return metrics


