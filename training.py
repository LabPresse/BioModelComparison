
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
        batch_size=16, n_epochs=50, lr=1e-3,
        segmentation=True, autoencoder=False, sigma=0.1,
        verbose=True, plot=True
    ):

    # Set up environment
    device = next(model.parameters()).device

    # Print status
    if verbose:
        status = ' '.join([
            f'Training model',
            f'with {sum(p.numel() for p in model.parameters())} parameters',
            f'on {device}.',
        ])
        print(status)

    # Track training stats
    epoch_times = []
    train_losses = []
    val_losses = []
    min_val_loss = float('inf')

    # Set up data loaders
    dataset_train, dataset_val, _ = datasets
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, 
        shuffle=True, num_workers=os.cpu_count(), pin_memory=True
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=batch_size, 
        shuffle=False, num_workers=os.cpu_count(), pin_memory=True
    )

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
            t_batch = time.time()  # Get batch time

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
            if verbose and (
                    (i == 0)                                   # Print first
                    or (i == len(dataloader_train) - 1)        # Print last
                    or ((i + 1) % 50 == 0)                     # Print every 50
                    or (len(dataloader_train) < 20)            # Print all if small dataset
                ):
                status = ' '.join([                            # Set up status
                    f'--',                                     # Indent
                    f'Batch {i+1}/{len(dataloader_train)}',    # Batch number
                    f'({(time.time()-t_batch):.2f} s/batch)',  # Time per batch
                ])
                print(status)  # Print status


        # Get validation loss
        if verbose:
            print('Validating')
        total_val_loss = 0                                     # Initialize loss
        for i, batch in enumerate(dataloader_val):             # Iterate over batches
            t_batch = time.time()                              # Get batch time
            x, y = get_batch(batch)                            # Get batch
            output = model(x)                                  # Get output
            loss = criterion(output, y)                        # Calculate loss
            total_val_loss += loss.item()                      # Update loss
            if verbose and (
                    (i == 0)                                   # Print first
                    or (i == len(dataloader_val) - 1)          # Print last
                    or ((i + 1) % 50 == 0)                     # Print every 50
                    or (len(dataloader_val) < 20)              # Print all if small dataset
                ):
                status = ' '.join([                            # Set up status
                    f'--',                                     # Indent
                    f'Batch {i+1}/{len(dataloader_val)}',      # Batch number
                    f'({(time.time()-t_batch):.2f} s/batch)',  # Time per batch
                ])
                print(status)  # Print status
    
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
            status = ':::' + '\n:::'.join([             # Set up status
                f'Train loss: {total_train_loss:.4e}',  # Print training loss
                f'Val loss: {total_val_loss:.4e}',      # Print validation loss
                f'Time: {time.time()-t:.2f} sec.'       # Print time
            ])
            print(status)
        
        # Plot images
        if plot:
            batch = next(iter(dataloader_train))  # Get batch
            if segmentation:                      # Segmentation
                x, y = batch                      # -- x is images, y is masks
                z = model(x).argmax(dim=1)        # -- z is highest probability class
            elif autoencoder:                     # Autoencoder
                x, y = get_batch(batch)           # -- x is noisy images, y is clean images
                z = model(x)                      # -- z is denoised images
            plot_images(
                Images=x[:5], 
                Targets=y[:5], 
                Predictions=z[:5]
            )
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

