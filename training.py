
# Import libraries
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import local modules
from helper_functions import plot_images


# Define training function
def train_model(
        model, datasets, savepath,
        batch_size=32, n_epochs=50, lr=1e-3,
        segmentation=True, autoencoder=False, sigma=0.1,
        verbose=True, plot=True
    ):

    # Print status
    if verbose:
        status = f'Training model with {sum(p.numel() for p in model.parameters())} parameters.'
        print(status)

    # Set up environment
    device = next(model.parameters()).device

    # Track training stats
    epoch_times = []
    train_losses = []
    val_losses = []
    min_val_loss = float('inf')

    # Set up data loaders
    dataset_train, dataset_val, _ = datasets
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    # Set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    max_grad_norm = 1/lr  # Max parameter update is 1

    # Set up loss
    if segmentation:
        criterion = nn.CrossEntropyLoss()
    elif autoencoder:
        criterion = nn.MSELoss()

    # Get batch function
    def get_batch(batch):
        
        # Get batch
        x, y = batch

        # Convert to correct type
        if segmentation:
            x = x.float()
            y = y.long()
        elif autoencoder:
            x = x.float()
            y = x.float()
            x = x + sigma * torch.randn_like(x)

        # Send to device
        x = x.to(device)
        y = y.to(device)

        # Return
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
            t_batch = time.time()

            # Zero gradients
            optimizer.zero_grad()

            # Get batch
            x, y = get_batch(batch)
            output = model(x)

            # Calculate loss
            loss = criterion(output, y)

            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Update loss
            total_train_loss += loss.item()

            # Print status
            if verbose and ((i % 10 == 0) or len(dataloader_train) < 20):
                status = ' '.join([
                    f'--',
                    f'Batch {i+1}/{len(dataloader_train)}',
                    f'({(time.time()-t_batch):.2f} s/batch)',
                ])
                print(status)


        # Get validation loss
        if verbose:
            print('Validating')
        total_val_loss = 0
        for i, batch in enumerate(dataloader_val):
            if verbose and ((i % 10 == 0) or len(dataloader_val) < 20):
                print(f'--Val Batch {i+1}/{len(dataloader_val)}')
            x, y = get_batch(batch)
            output = model(x)
            loss = criterion(output, y)
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
            status = ':::' + '\n:::'.join([
                f'Train loss: {total_train_loss:.4e}',
                f'Val loss: {total_val_loss:.4e}',
                f'Time: {time.time()-t:.2f} sec.'
            ])
            print(status)
        
        # Plot images
        if plot:
            batch = next(iter(dataloader_train))
            if segmentation:
                x, y = batch
                x = x.float().to(device)
                y = y.long().to(device)
                z = model(x).argmax(dim=1)
            elif autoencoder:
                x, y = get_batch(batch)
                z = model(x)
            plot_images(Images=x[:5], Targets=y[:5], Predictions=z[:5])
            plt.pause(1)

    # Load best model
    model.load_state_dict(torch.load(savepath))

    # Finalize training stats
    statistics = {
        'n_parameters': int(sum(p.numel() for p in model.parameters())),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'min_val_loss': min_val_loss,
        'epoch_times': epoch_times,
    }
    
    # Return model
    if verbose:
        print('Training complete.')
    return model, statistics

