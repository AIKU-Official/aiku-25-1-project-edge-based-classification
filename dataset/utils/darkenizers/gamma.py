from PIL import Image
import numpy as np

def gamma_correction(image, gamma=3.0):
    # Convert to numpy array
    img_array = np.array(image) / 255.0  # Normalize to [0, 1]
    img_array = np.power(img_array, gamma)  # Apply gamma correction
    img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)  # Rescale to [0, 255]
    
    return Image.fromarray(img_array)