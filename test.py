
# Import libraries
import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk

def enlarge_blob(mask, dot_radius=5):
    """Enlarge the blob in the mask."""

    # Get blob
    blob = np.where(mask > 0)
    blob = np.array([blob[0], blob[1]]).T

    # Get random point
    for point in blob:
        # Enlarge blob
        rr, cc = disk(point, dot_radius, shape=mask.shape)
        mask[rr, cc] = 1

    # Return mask
    return mask


# Get files
root = os.path.join('data', 'bdello')
files = os.listdir(os.path.join(root, 'images'))
files = [f for f in files if f.endswith('.png')]

# Set up a figure
fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
plt.ion()
plt.show()

# Loop over files
for file in files:
        
    # Open mask
    img = Image.open(os.path.join(root, 'images', file))
    mask = Image.open(os.path.join(root, 'masks', file))

    # Convert to numpy
    img = np.array(img)
    mask = np.array(mask)

    # # Enlarge blob
    # mask = enlarge_blob(mask)
    # # Convert to rgb
    # mask = 255*mask  # np.stack([mask, mask, mask], axis=2)

    # # Pad image to square
    # max_side = max(img.shape[0], img.shape[1])
    # pad_left = (max_side - img.shape[1]) // 2
    # pad_right = max_side - img.shape[1] - pad_left
    # pad_top = (max_side - img.shape[0]) // 2
    # pad_bottom = max_side - img.shape[0] - pad_top
    # img = np.pad(img, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
    # if pad_left+pad_right+pad_top+pad_bottom > 0:
    #     print(f'Padded {file} image with {pad_left}, {pad_right}, {pad_top}, {pad_bottom} pixels')

    # # Pad mask to square
    # max_side = max(mask.shape[0], mask.shape[1])
    # pad_left = (max_side - mask.shape[1]) // 2
    # pad_right = max_side - mask.shape[1] - pad_left
    # pad_top = (max_side - mask.shape[0]) // 2
    # pad_bottom = max_side - mask.shape[0] - pad_top
    # mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
    # if pad_left+pad_right+pad_top+pad_bottom > 0:
    #     print(f'Padded {file} mask with {pad_left}, {pad_right}, {pad_top}, {pad_bottom} pixels')

    # # Save as png
    # img = Image.fromarray(img)
    # img.save(os.path.join(root, 'images', file))
    # mask = Image.fromarray(mask)
    # mask.save(os.path.join(root, 'masks', file))

    # Show
    ax[0].cla(); ax[1].cla()
    ax[0].imshow(img)
    ax[1].imshow(mask)
    plt.pause(0.1)

