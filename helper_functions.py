
# Import libraries
import math
import torch
import numpy as np
import matplotlib.pyplot as plt


# Get savename
def get_savename(modelID, dataID, options, ffcvid=0):
    savename = f'{modelID}_{dataID}'
    for key in sorted(list(options.keys())):
        savename += f'_{key}={options[key]}'
    savename += f'_ffcv={ffcvid}'
    return savename


# Convert to serializable function
def convert_to_serializable(item):
    if isinstance(item, dict):
        return {k: convert_to_serializable(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [convert_to_serializable(v) for v in item]
    elif isinstance(item, tuple):
        return tuple(convert_to_serializable(v) for v in item)
    elif isinstance(item, set):
        return {convert_to_serializable(v) for v in item}
    elif isinstance(item, np.generic):
        return item.item()
    elif isinstance(item, np.ndarray):
        return convert_to_serializable(item.tolist())
    elif isinstance(item, torch.Tensor):
        return convert_to_serializable(item.cpu().detach().numpy())
    else:
        return item


# Count trainable parameters
def count_parameters(model, verbose=False):
    """Count the number of trainable parameters in a model."""
    if verbose:
        print("Model layers and their parameter counts:")

    # Initialize total parameters
    total_params = 0

    # Loop over model parameters
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:  # Only counting trainable parameters
            param_count = parameter.numel()  # Number of elements in the parameter
            total_params += param_count
            if verbose:
                print(f"{name}: {param_count}")
    
    # Done
    if verbose:
        print(f"Total trainable parameters: {total_params}")
    return total_params


# Create check gradient function
def check_gradient(model, verbose=False):

    # Initialize gradient statistics
    max_grad = -torch.inf
    min_grad = torch.inf 
    mean_grad = 0
        
    # Get gradient statistics
    layer_maxes = []
    for name, p in model.named_parameters():
        if p.grad is not None:

            # Get max
            max_grad_p = p.grad.abs().max().item()
            layer_maxes.append(max_grad_p)
            if max_grad_p > max_grad:
                max_grad = max_grad_p
                max_grad_name = name

            # Get min
            min_grad_p = p.grad.abs().min().item()
            if min_grad_p < min_grad:
                min_grad = min_grad_p
                min_grad_name = name
            
            # Get mean
            mean_grad += p.grad.abs().sum().item()
            layer_maxes.append(p.grad.abs().max().item())

    # Normalize mean
    mean_grad /= sum(p.numel() for p in model.parameters())

    # Print gradient statistics
    if verbose:
        print(f"Max grad at {max_grad_name} = {max_grad:.4e}")
        print(f"Min grad at {min_grad_name} = {min_grad:.4e}")
        print(f"Mean grad = {mean_grad:.4e}")

    # Return gradient statistics
    return max_grad, min_grad, mean_grad, layer_maxes


# Check parameters function
def check_parameters(model, verbose=False):

    # Get parameter statistics
    max_p = max(p.abs().max().item() for p in model.parameters())
    min_p = min(p.abs().min().item() for p in model.parameters())
    mean_p = (
        sum(p.abs().sum().item() for p in model.parameters())
        / sum(p.numel() for p in model.parameters())
    )

    # Find layerwise max parameters
    layer_maxes = []
    for name, p in model.named_parameters():
        layer_maxes.append(p.abs().max().item())
        # Get max layer
        if p.abs().max().item() == max_p:
            max_p_name = p
        # Get min layer
        if p.abs().min().item() == min_p:
            min_p_name = p

    # Print parameter statistics
    if verbose:
        print(f"Max parameter = {max_p:.4e} at {max_p_name}")
        print(f"Min parameter = {min_p:.4e} at {min_p_name}")
        print(f"Mean parameter = {mean_p:.4e}")
    
    # Return parameter statistics
    return max_p, min_p, mean_p, layer_maxes


# Check tensor function
def check_tensor(x, verbose=False):

    # Get tensor statistics
    max_x = x.abs().max().item()
    min_x = x.abs().min().item()
    mean_x = x.abs().mean().item()
    
    # Print tensor statistics
    if verbose:
        print(f"Max tensor = {max_x:.4e}")
        print(f"Min tensor = {min_x:.4e}")
        print(f"Mean tensor = {mean_x:.4e}")

    # Return tensor statistics
    return max_x, min_x, mean_x


# Plot tensors function
@torch.no_grad()
def plot_images(images=None, col_labels=None, **image_dict):

    # Set up image_dict
    if images is not None:
        image_dict['images'] = images
    num_arrays = len(image_dict.keys())

    # Send tensors to cpu and numpy
    for key in image_dict.keys():
        val = image_dict[key]
        if isinstance(val, torch.Tensor):
            image_dict[key] = image_dict[key].float().cpu().detach().numpy()
        elif isinstance(val, np.ndarray):
            image_dict[key] = image_dict[key].astype(float)

    # Assert that all image arrays have the same number of images
    num_images = image_dict[list(image_dict.keys())[0]].shape[0]
    for key in image_dict.keys():
        if image_dict[key].shape[0] != num_images:
            raise ValueError("All image arrays must have the same number of images.")

    # Set up colunm labels
    if col_labels is None:
        col_labels = [f'Image {i}' for i in range(num_images)]

    # Get figure
    fig = plt.gcf()
    fig.set_size_inches(num_images, num_arrays)
    plt.clf()
    plt.ion()
    plt.show()
    ax = np.empty((num_arrays, num_images), dtype=object)
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i, j] = fig.add_subplot(ax.shape[0], ax.shape[1], i * ax.shape[1] + j + 1)

    # Loop over image lists
    for i, (key, val) in enumerate(image_dict.items()):

        # Set label
        ax[i, 0].set_ylabel(key)

        # Loop over images
        for j in range(val.shape[0]):

            # Get image
            img = val[j]

            # Slice batch if necessary
            if len(img.shape) > 3:
                img = img[0]

            # Pad image channels if necessary
            if img.shape[0] == 1:
                img = img[0]
            elif img.shape[0] == 2:
                img = np.concatenate((img, np.zeros((1,*img[0].shape))), axis=0).transpose(1, 2, 0)
            elif img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))

            # Normalize image
            img = img - img.min()
            if img.max() > 0:
                img = img / img.max()

            # Plot image
            ax[i, j].imshow(img)

    # Finalize plot
    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[0, j].set_title(col_labels[j])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.tight_layout()
    plt.pause(1)
    
    # Return
    return fig, ax


# Plor ROC curve function
def plot_roc_curve(fpr, tpr, accuracy=None, sensitivity=None, specificity=None, auc_score=None):
    """Plot the ROC curve with the AUC score."""
    
    # Set up figure
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(6, 6)
    plt.ion()
    plt.show()

    # Set up label
    label = 'ROC curve'
    if auc_score is not None:
        auc_score = round(auc_score, 2)
        label = label + f' (AUC = {auc_score})'
    if accuracy is not None:
        accuracy = round(accuracy, 2)
        label = label + f' (Accuracy = {accuracy})'
    if sensitivity is not None:
        sensitivity = round(sensitivity, 2)
        label = label + f' (Sensitivity = {sensitivity})'

    # Plot ROC curve
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
    ax.plot(fpr, tpr, color='blue', label=label)

    # Set up plot
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend()
    plt.pause(.1)
    
    # Return
    return fig, ax
