import random
import numpy as np
from PIL import Image

def dark_image_with_noise(image, noise_factor=0.5):
    img_array = np.array(image)
    noise = np.random.normal(scale=noise_factor, size=img_array.shape)  # Add noise
    darkened_image = np.clip(img_array * 0.3 + noise, 0, 255).astype(np.uint8)  # Darken and add noise
    return Image.fromarray(darkened_image)