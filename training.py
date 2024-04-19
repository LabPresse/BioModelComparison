
# Import libraries
import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import local modules
from helper_functions import plot_images
from testing import evaluate_model


# Define training function
def train_model(model, datasets, savepath,
    segmentation=False, autoencoder=False, mae=False, dae=False,
    batch_size=32, n_epochs=50, lr=1e-3, num_workers=0,
    verbose=True, plot=True
    ):

    # Check inputs
    if not (segmentation or autoencoder):
        raise ValueError('Must specify either segmentation or autoencoder.')

    # Print status
    if verbose:
        status = f'Training model with {sum(p.numel() for p in model.parameters())} parameters.'
        print(status)

    # Set up environment
    device = next(model.parameters()).device

    # Track training stats
    train_losses = []
    val_losses = []
    min_val_loss = float('inf')
    epoch_times = []

    # Set up data loaders
    dataset_train, dataset_val, _ = datasets
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    max_grad_norm = 1/lr  # Max parameter update is 1

    # Set up criterion
    if segmentation:
        loss_function = torch.nn.CrossEntropyLoss()
    elif autoencoder:
        loss_function = torch.nn.MSELoss()

    # Define extract batch function
    def extract_batch(batch):
        if segmentation:
            x, y = batch
            x = x.float()
            y = y.long()
        elif autoencoder:
            x, _ = batch
            x = x.to(device).float()
            y = x.clone()
            if dae:
                x = x + .1 * torch.randn_like(x)
            if mae:
                mask = model.create_mask(x.shape[0])
                x = x * mask
        x = x.to(device)
        y = y.to(device)
        return x, y

    # Train model
    for epoch in range(n_epochs):
        t = time.time()
        if verbose:
            print(f'Epoch {epoch+1}/{n_epochs}')

        # Initialize loss
        total_train_loss = 0

        # Iterate over batches
        for i, batch in enumerate(dataloader_train):
            if verbose and ((i % 10 == 0) or len(dataloader_train) < 20):
                print(f'--Batch {i+1}/{len(dataloader_train)}')

            # Zero gradients
            optimizer.zero_grad()

            # Calculate loss
            x, y = extract_batch(batch)
            output = model(x)
            loss = loss_function(output, y)

            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Update loss
            total_train_loss += loss.item()


        # Get validation loss
        if verbose:
            print('Validating')
        total_val_loss = 0
        for i, batch in enumerate(dataloader_val):
            if verbose and ((i % 10 == 0) or len(dataloader_val) < 20):
                print(f'--Val Batch {i+1}/{len(dataloader_val)}')
            x, y = extract_batch(batch)
            output = model(x)
            loss = loss_function(output, y)
            total_val_loss += loss.item()
    
        # Save model if validation loss is lower
        if total_val_loss < min_val_loss:
            min_val_loss = total_val_loss
            torch.save(model.state_dict(), savepath)

        # Save training metrics
        train_losses.append(total_train_loss / len(dataset_train))
        val_losses.append(total_val_loss / len(dataset_val))
        epoch_times.append(time.time()-t)

        # Print status
        if verbose:
            status = '::' + '::\n'.join([
                f'Train loss: {total_train_loss:.4e}',
                f'Val loss: {total_val_loss:.4e}',
                f'Time: {time.time()-t:.2f} sec.'
            ])
            print(status)
        
        # Plot images
        if plot:
            x, y = extract_batch(next(iter(dataloader_train)))
            z = model(x)
            if segmentation:
                z = model(x).argmax(dim=1)
            plot_images(Images=x[:5], Predictions=z[:5], Targets=y[:5])
            plt.pause(1)

    # Load best model
    model.load_state_dict(torch.load(savepath))

    # Finalize training stats
    statistics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'min_val_loss': min_val_loss,
        'epoch_times': epoch_times,
    }
    
    # Return model
    if verbose:
        print('Training complete.')
    return model, statistics

