from PIL import ImageOps

def inverse_histogram_equalization(image):
    # Convert to grayscale
    gray_image = image.convert("L")
    # Apply inverse histogram equalization
    inverted_image = ImageOps.invert(gray_image)
    return inverted_image.convert("RGB")  # Convert back to RGB