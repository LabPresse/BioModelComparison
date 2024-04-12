
# Import libraries
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


# Define a function to evaluate a PyTorch model
def evaluate_model(model, dataloader):
    """Evaluate a PyTorch model on a dataset."""

    # Set the device
    device = next(model.parameters()).device

    # Set the model to evaluation mode
    model.eval()

    # Lists to store true labels and predicted probabilities
    true_labels = []
    predicted_probs = []

    # Iterate over the dataloader
    for x, y in dataloader:

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

    # Calculate the predicted labels based on the predicted probabilities
    predicted_labels = np.argmax(predicted_probs, axis=1)

    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

    # Calculate sensitivity, specificity, and accuracy
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Calculate the ROC curve and AUC score
    roc_fpr, roc_tpr, roc_thresholds = roc_curve(true_labels, predicted_probs[:, 1])
    auc_score = roc_auc_score(true_labels, predicted_probs[:, 1])

    # Package the metrics in a dictionary
    metrics = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'roc_fpr': roc_fpr,
        'roc_tpr': roc_tpr,
        'roc_thresholds': roc_thresholds,
        'auc_score': auc_score
    }

    # Return the evaluation metrics
    return metrics
