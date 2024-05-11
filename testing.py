
# Import libraries
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# Import local modules
from helper_functions import plot_roc_curve


# Define a function to evaluate a PyTorch model
def evaluate_model(model, dataset, batch_size=32, verbose=True, plot=False):
    """Evaluate a PyTorch model on a dataset."""

    ### EVALUATE THE MODEL ###
    if verbose:
        status = f'Evaluating model with {sum(p.numel() for p in model.parameters())} parameters.'
        print(status)

    # Set up model and device
    model.eval()
    device = next(model.parameters()).device

    # Create lists to store true labels and predicted probabilities
    true_labels = []
    predicted_probs = []

    # Set up the dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Iterate over the dataloader
    for i, (x, y) in enumerate(dataloader):
        if verbose and ((i % 10 == 0) or (len(dataloader) < 20)):
            print(f'--Batch {i + 1}/{len(dataloader)}')

        # Move the data to the device
        x = x.to(device)
        y = y.to(device)

        # Forward pass through the model
        z = model(x)
        probs = torch.softmax(z, dim=1)

        # Append the predicted probabilities and true labels to the lists
        predicted_probs.extend(probs.detach().numpy())
        true_labels.extend(y.numpy())

    # Convert the lists to numpy arrays
    true_labels = np.array(true_labels)
    predicted_probs = np.array(predicted_probs)


    ### CALCULATE EVALUATION METRICS ###
    if verbose:
        print('Calculating evaluation metrics.')

    # Calculate the confusion matrix
    predicted_labels = np.argmax(predicted_probs, axis=1)
    tn, fp, fn, tp = confusion_matrix(
        true_labels.reshape(-1), 
        predicted_labels.reshape(-1)
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
        true_labels.reshape(-1),
        np.round(predicted_probs[:, 1].reshape(-1), 3),  # Round input to avoid too many thresholds
    )
    auc_score = roc_auc_score(
        true_labels.reshape(-1),
        np.round(predicted_probs[:, 1].reshape(-1), 3),  # Round input to avoid too many thresholds
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


