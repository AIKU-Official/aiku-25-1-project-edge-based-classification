from PIL import Image, ImageEnhance

def naive_darkenizer(img, brightness_scale=0.3, bluelight=40):
    # Darken image (reduce brightness)
    enhancer = ImageEnhance.Brightness(img)
    img_dark = enhancer.enhance(brightness_scale)  # 0.3 = 30% brightness

    # Optionally: add blue tint
    # blue_overlay = Image.new('RGB', img.size, (0, 0, bluelight))
    # img_night = Image.blend(img_dark, blue_overlay, alpha=0.3)

    return img_dark