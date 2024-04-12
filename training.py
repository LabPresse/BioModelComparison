
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
    segmentation=False, autoencoder=False, mae=False,
    batch_size=32, n_epochs=1000, lr=1e-3,
    verbose=True, plot=True, device=torch.device('cpu'),
    ):

    # Print status
    if verbose:
        status = f'Training model with {sum(p.numel() for p in model.parameters())} parameters.'
        print(status)

    # Check inputs
    if not (segmentation or autoencoder):
        raise ValueError('Must specify either segmentation or autoencoder.')

    # Track training stats
    train_losses = []
    val_losses = []
    min_val_loss = float('inf')
    epoch_times = []

    # Set up data loaders
    dataset_train, dataset_val, _ = datasets
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    # dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # Set up model and optimizer
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    max_grad_norm = 1

    # Set up criterion
    if segmentation:
        loss_function = torch.nn.CrossEntropyLoss()
    elif autoencoder:
        loss_function = torch.nn.MSELoss()

    # Define calculate loss function
    def calculate_loss(model, batch):
            
        # Extract data
        if segmentation:
            x, y = batch
            x = x.to(device).float()
            y = y.to(device).long()
        elif autoencoder:
            x, _ = batch
            x = x.to(device).float()
            y = x.clone()
            if mae:
                mask = model.create_mask(x.shape[0]).to(device)
                x = x * mask

        # Forward pass
        output = model(x)

        # Calculate loss
        loss = loss_function(output, y)

        # Return loss
        return loss

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
            loss = calculate_loss(model, batch)

            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            # Update loss
            total_train_loss += loss.item()


        # Get validation loss
        if verbose:
            print('--Validating')
        total_val_loss = 0
        for i, batch in enumerate(dataloader_val):
            if verbose and ((i % 10 == 0) or len(dataloader_val) < 20):
                print(f'--Val Batch {i+1}/{len(dataloader_val)}')
            loss = calculate_loss(model, batch)
            total_val_loss += loss.item()
    
        # Save model if validation loss is lower
        if total_val_loss < min_val_loss:
            min_val_loss = total_val_loss
            torch.save(model.state_dict(), savepath)

        # Save training metrics
        train_losses.append(total_train_loss)
        val_losses.append(total_val_loss)
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
            x, y = next(iter(dataloader_val))
            x = x.to(device).float()
            y = y.to(device).long()
            if segmentation:
                z = model(x).argmax(dim=1)
            elif autoencoder:
                y = x.clone()
                if mae:
                    mask = model.create_mask(x.shape[0]).to(device)
                    x = x * mask
                z = model(x)
            plot_images(Images=x[:5], Predictions=z[:5], Targets=y[:5])
            plt.pause(1)

    # Load best model
    model.load_state_dict(torch.load(savepath))

    # # Test model
    # if verbose:
    #     print('Testing model.')
    # test_metrics = evaluate_model(model, dataloader_test)

    # Finalize training stats
    statistics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'min_val_loss': min_val_loss,
        'epoch_times': epoch_times,
        # 'test_metrics': test_metrics
    }
    
    # Return model
    if verbose:
        print('Training complete.')
    return model, statistics

