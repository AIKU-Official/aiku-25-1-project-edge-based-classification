from PIL import Image, ImageEnhance
import os
import argparse
from tqdm import tqdm
from utils.darkenizers.naive import naive_darkenizer
from utils.darkenizers.gamma import gamma_correction
from utils.darkenizers.HSV import hsv_manipulation
from utils.darkenizers.histogram import inverse_histogram_equalization
from utils.darkenizers.noise import dark_image_with_noise

parser = argparse.ArgumentParser()

parser.add_argument("--method", default="naive", type=str,
                    help="Option : [HSV, gamma, naive, histogram, noise]")
parser.add_argument("--src-dir", default="./imagenet-1k-subset", type=str,
                    help="Source directory to apply darkenizing")
parser.add_argument("--dst-dir", default="./Dark-ImageNet-subset", type=str,
                    help="Destination directory to save darkenized images")
parser.add_argument("--dark-opt", default="default", type=str,
                    help="Darkenizer option")

args = parser.parse_args()

# === Set darkening method here ===
darkening_method = args.method  # options: "HSV", "gamma", "naive", "histogram", "noise"

# === Map method name to function ===
darkenizer_fn = {
    "HSV": hsv_manipulation,
    "gamma": gamma_correction,
    "naive": naive_darkenizer,
    "histogram": inverse_histogram_equalization,
    "noise": dark_image_with_noise
}[darkening_method]

# src_dir = "./ImageNet-S/ImageNetS50/validation"
# dark_dataset_dir = "./Dark-ImageNet-S50"
src_dir = args.src_dir
dark_dataset_dir = args.dst_dir
output_dir = f"{dark_dataset_dir}/{darkening_method}-{args.dark_opt}"

os.makedirs(output_dir, exist_ok=True)

def darkenize_folder(src_dir, split="val"):
    src_dir = os.path.join(src_dir, split)
    
    for folder in tqdm(os.listdir(src_dir)):
        folder_path = os.path.join(src_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        output_subfolder = os.path.join(output_dir, split, folder)
        os.makedirs(output_subfolder, exist_ok=True)

        for image_name in os.listdir(folder_path):
            # Load image
            img = Image.open(f'{folder_path}/{image_name}').convert('RGB')
            
            if args.dark_opt != "default":
                img_night = darkenizer_fn(img, float(args.dark_opt))
            else:
                img_night = darkenizer_fn(img)

            img_night.save(f'{output_subfolder}/{image_name}')
            
darkenize_folder(src_dir, "train")
darkenize_folder(src_dir, "val")
darkenize_folder(src_dir, "test")