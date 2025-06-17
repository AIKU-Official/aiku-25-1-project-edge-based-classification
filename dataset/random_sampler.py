import os
import random
import shutil
from tqdm import tqdm

# Parameters
src_dir = "./imagenet-1k/val"
output_dir = "./imagenet-1k-subset/val"
num_samples = 5

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Loop through folders in current directory
for folder in tqdm(os.listdir(src_dir)):
    folder_path = os.path.join(src_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    
    # Get .JPEG files only (case-sensitive)
    image_files = [f for f in os.listdir(folder_path) if f.endswith(".JPEG")]

    # Randomly sample up to 5 images
    sampled_files = random.sample(image_files, min(num_samples, len(image_files)))

    # Create subfolder in output directory
    output_subfolder = os.path.join(output_dir, folder)
    os.makedirs(output_subfolder, exist_ok=True)

    # Copy sampled images
    for file in sampled_files:
        src = os.path.join(folder_path, file)
        dst = os.path.join(output_subfolder, file)
        shutil.copy2(src, dst)

print("Sampling completed.")