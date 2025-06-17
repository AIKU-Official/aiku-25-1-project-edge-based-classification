from PIL import Image, ImageEnhance
import numpy as np

def hsv_manipulation(image, value_factor=0.3):
    # Convert to HSV
    hsv_image = image.convert("HSV")
    hsv_array = np.array(hsv_image).astype(np.float32)

    # Scale the V (brightness) channel
    hsv_array[..., 2] *= value_factor
    hsv_array[..., 2] = np.clip(hsv_array[..., 2], 0, 255)

    # Convert back to PIL image
    hsv_image_dark = Image.fromarray(hsv_array.astype(np.uint8), mode="HSV").convert("RGB")
    return hsv_image_dark