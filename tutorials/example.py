

# Import libraries
import os
import sys
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset, random_split

# Set environment
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


######################
### CREATE DATASET ###
######################

# Create an example datset class
class MyDataset(Dataset):
    """
    This is a simple dataset class that simply loads noisy images of squares.
    The intended purpose of this class is to provide a simple example of how to
    create a custom dataset class in PyTorch. Feel free to modify this class to
    suit your specific use case needs.
    """
    def __init__(self, shape=(128, 128), square_size=16, n_samples=1000):
        super(MyDataset, self).__init__()
        self.shape = shape
        self.n_samples = n_samples
        self.square_size = square_size
        

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):

        # Create mask
        mask = torch.zeros(*self.shape)

        # Randomly insert a square into the mask
        x = torch.randint(0, self.shape[0] - self.square_size, (1,))  # x-coordinate
        y = torch.randint(0, self.shape[1] - self.square_size, (1,))  # y-coordinate
        mask[x:x+self.square_size, y:y+self.square_size] = 1

        # Create noisy image
        image = mask + 0.1 * torch.randn(*self.shape)

        # Format image and mask
        mask = mask.long()
        image = image.unsqueeze(0)

        # Return
        return image, mask



####################
### CREATE MODEL ###
####################

# Define a simple convolutional model class
class MyModel(nn.Module):
    def __init__(self, 
            in_channels, out_channels, 
            n_features=8, n_layers=16
        ):
        super(MyModel, self).__init__()

        # Set up attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_features = n_features
        self.n_layers = n_layers


        ### SET UP BLOCKS ###

        # Set up input block
        self.input_block = nn.Sequential(
            nn.GroupNorm(1, in_channels, affine=False),  # Normalize input
            nn.Conv2d(in_channels, n_features, kernel_size=3, padding=1),
        )

        # Set up layers
        layers = []
        for i in range(n_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv2d(n_features, n_features, kernel_size=3, padding=1),
                    nn.InstanceNorm2d(n_features),
                    nn.ReLU(),
                )
            )
        self.layers = nn.Sequential(*layers)

        # Set up output block
        self.output_block = nn.Sequential(
            nn.Conv2d(self.n_features, out_channels, kernel_size=1),
        )

    def forward(self, x):
        """Forward pass."""

        # Input block
        x = self.input_block(x)

        # Layers
        x = self.layers(x)

        # Output block
        x = self.output_block(x)

        # Return
        return x


################################
### CREATE TRAINING FUNCTION ###
################################

# Define a simple training function
def train_fn(model, dataset_train, dataset_val=None, n_epochs=10):
    """Train the model."""

    # Set up model
    model.to(device)
    model.train()

    # Set up dataloaders
    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
    if dataset_val is not None:
        dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=False)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Set up loss function for segmentation
    def loss_fn(x, y):
        
        # Calculate loss
        loss = F.cross_entropy(x, y)  # Segmentation loss
        # loss = F.mse_loss(x, y)  # Autoencoder loss

        # Regularization loss
        loss += (
            sum(p.pow(2.0).sum() for p in model.parameters()) 
            / sum(p.numel() for p in model.parameters())
        )

        # Return
        return loss
    
    # Set up best model
    if dataset_val is not None:
        best_model = copy.deepcopy(model)
        best_val_loss = float('inf')
    
    # Train model
    print('Training model')
    for epoch in range(n_epochs):

        # Loop over training dataloader
        total_loss = 0
        for i, (x, y) in enumerate(dataloader_train):
            
            # Configure batch
            x = x.to(device)
            y = y.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(x)

            # Compute loss
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        # Validate model if validation dataset provided
        total_val_loss = 0
        if dataset_val is not None:
            with torch.no_grad():
                for i, (x, y) in enumerate(dataloader_val):
                    x = x.to(device)
                    y = y.to(device)
                    y_pred = model(x)
                    loss = loss_fn(y_pred, y)
                    total_val_loss += loss.item()
            if total_val_loss <= best_val_loss:
                best_val_loss = total_val_loss
                best_model = copy.deepcopy(model)

        # Print loss
        print(f'Epoch {epoch+1}/{n_epochs} -- Train Loss: {total_loss:.2f} -- Val Loss: {total_val_loss:.2f}')

    # Get best model
    if dataset_val is not None:
        model = best_model

    # Return
    return model



#############################
### CREATE TEST FUNCTIONS ###
#############################

# Define visualization function
def visualize(x, y, y_pred, n_samples=5):
    """Visualize data."""

    # Set up figure
    fig, ax = plt.subplots(n_samples, 3, figsize=(12, 3*n_samples))
    fig.set_size_inches(9, 3*n_samples)
    plt.ion()
    plt.show()

    # Plot samples
    for i in range(n_samples):
        ax[i, 0].set_ylabel(f'Sample {i+1}')
        ax[i, 0].set_title('Input')
        ax[i, 0].imshow(x[i].squeeze().cpu().numpy(), cmap='gray')
        ax[i, 0].axis('off')
        ax[i, 1].set_title('Target')
        ax[i, 1].imshow(y[i].squeeze().cpu().numpy(), cmap='gray')
        ax[i, 1].axis('off')
        ax[i, 2].set_title('Prediction')
        ax[i, 2].imshow(y_pred[i].squeeze().cpu().numpy(), cmap='gray')

    # Finalize
    plt.tight_layout()
    plt.pause(1)

    # Return
    return fig, ax


# Define test function
def test_fn(model, dataset_test):
    """
    This model evaluates the model on a test dataset and returns the
    accuracy, sensitivity, and specificity of the model.
    """
    
    # Set up dataloader
    dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=False)

    # Set up metrics
    total_accuracy = 0
    total_sensitivity = 0
    total_specificity = 0

    # Loop over test dataset
    for i, (x, y) in enumerate(dataloader_test):
        
        # Configure batch
        x = x.to(device)
        y = y.to(device)

        # Forward pass
        y_pred = model(x)
        y_pred = y_pred.argmax(dim=1)

        # Compute metrics
        tp = (y_pred * y).sum().item()
        tn = ((1 - y_pred) * (1 - y)).sum().item()
        fp = (y_pred * (1 - y)).sum().item()
        fn = ((1 - y_pred) * y).sum().item()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        # Update metrics
        total_accuracy += accuracy
        total_sensitivity += sensitivity
        total_specificity += specificity

    # Compute final metrics
    n_samples = len(dataset_test)
    total_accuracy /= n_samples
    total_sensitivity /= n_samples
    total_specificity /= n_samples

    # Return
    return {
        'accuracy': total_accuracy,
        'sensitivity': total_sensitivity,
        'specificity': total_specificity,
    }



###################
### MAIN SCRIPT ###
###################

# Main
if __name__ == '__main__':

    # Set up dataset
    dataset = MyDataset(n_samples=100)
    dataset_train, dataset_val, dataset_test = random_split(dataset, [60, 20, 20])

    # Set up model
    model = MyModel(1, 2)

    # Train model
    model = train_fn(model, dataset_train, dataset_val)

    # Test model
    metrics = test_fn(model, dataset_test)
    print(metrics)

    # Viziualize model
    dataloader_test = DataLoader(dataset_test, batch_size=16, shuffle=False)
    x, y = next(iter(dataloader_test))
    y_pred = model(x.to(device)).argmax(dim=1)
    visualize(x, y, y_pred)

    # # Save model  Uncomment to save model
    # savepath = 'model.pth'
    # torch.save(model.state_dict(), savepath)

    # Done
    print('Done.')

