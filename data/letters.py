
# Import libraries
import torch
import unicodedata
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, RandomAffine, GaussianBlur

        
# Letters dataset
class LettersDataset(Dataset):
    """Letters dataset."""
    
    def __init__(self, shape=(64, 64), sigma=.5, blur=None, transform=True, num_letters=1000):
        super().__init__()

        # Generate a list of all the Latin Letters
        letters = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            # 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
            # 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        ]

        # Repeat letters to get the desired number of letters
        letters = letters * (num_letters // len(letters) + 1)
        letters = letters[:num_letters]

        # Configure values
        if blur is None:
            blur = .01*max(shape)
        
        # Set the parameters
        self.shape = shape
        self.transform = transform
        self.letters = letters
        self.num_letters = len(letters)
        self.font_path = "data/fonts/NotoSansSC-Regular.otf"
        self.sigma = sigma
        self.blur = blur

    def __len__(self):
        return self.num_letters

    def __getitem__(self, idx):
        
        # Get the letter
        letter = self.letters[idx]

        # Get the image
        image = self.print_letter(letter)

        # Distort the image
        if self.transform:
            image = RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.8, 1),
                shear=(0.1, 0.1),
                fill=1
            )(image)
            image = GaussianBlur(kernel_size=9, sigma=.01*max(self.shape))(image)  # Small blur for stability
        
        # Get mask
        mask = image[0, :, :] < 0.5
        mask = mask.long()
        
        # Add noise
        image = GaussianBlur(kernel_size=9, sigma=self.blur)(image)
        image += self.sigma * torch.randn_like(image)

        # Finalize the image
        image -= image.min()
        if image.max() > 0:
            image /= image.max()

        return image, mask

    def _get_font_size(self, text, font_ratio):

        # Calculate the maximum font size based on the image size
        max_font_size = int(max(self.shape) * font_ratio)

        # Load the font file
        font = ImageFont.truetype(self.font_path, size=max_font_size)

        # Calculate the size of the text with the maximum font size
        text_bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), text, font)
        text_size = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        # Calculate the font size that fits within the image
        font_size = int(font_ratio * max(self.shape) / max(text_size) * max_font_size)
        return font_size

    def print_letter(self, text, font_ratio=0.8):

        # Create an image
        image = Image.new('L', self.shape, color=255)

        # Get the font
        font_size = self._get_font_size(text, font_ratio)
        font = ImageFont.truetype(self.font_path, size=font_size)

        # Get the text size and location
        text_bbox = ImageDraw.Draw(Image.new('RGB', (1, 1))).textbbox((0, 0), text, font)
        text_size = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        ascent = int(font.getmetrics()[1] * font_size / text_size[1])
        x = (self.shape[0] - text_size[0]) // 2
        y = (self.shape[1] - text_size[1]) // 2 - ascent

        # Draw the text
        draw = ImageDraw.Draw(image)
        draw.text((x, y), text, font=font, fill=0)

        # Convert the image to a tensor
        image = ToTensor()(image)

        # Return the image
        return image


# Test the dataset
if __name__ == "__main__":

    # Create a dataset
    dataset = LettersDataset(
        shape=(128, 128),
        blur=16, sigma=2,
    )

    # Print some letters
    fig, ax = plt.subplots(1, 2)
    plt.ion()
    plt.show()
    for i in range(10):
        image, mask = dataset[i]
        ax[0].cla()
        ax[1].cla()
        ax[0].imshow(image[0, :, :].detach().numpy(), cmap='gray')
        ax[1].imshow(mask.detach().numpy(), cmap='gray')
        plt.pause(1)

    # Done
    print("Done.")
    