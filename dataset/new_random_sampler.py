import os
import random
import shutil
from tqdm import tqdm

# Parameters
src_dir = "./imagenet-1k/val"
output_root = "./imagenet-1k-subset-v2"
split_ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
num_samples_per_class = 20  # 클래스별 총 샘플 수 (train+val+test 합)

# Make sure ratios sum to 1.0
assert abs(sum(split_ratios.values()) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

# Prepare output directories
for split in split_ratios:
    os.makedirs(os.path.join(output_root, split), exist_ok=True)

# Loop through class folders
for class_name in tqdm(os.listdir(src_dir), desc="Processing classes"):
    class_path = os.path.join(src_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    # Get .JPEG files
    image_files = [f for f in os.listdir(class_path) if f.endswith(".JPEG")]
    random.shuffle(image_files)

    # Sample fixed number per class
    image_files = image_files[:num_samples_per_class]
    n_total = len(image_files)

    # Calculate number per split
    n_train = int(split_ratios["train"] * n_total)
    n_val = int(split_ratios["val"] * n_total)
    n_test = n_total - n_train - n_val  # rest goes to test

    split_counts = {
        "train": n_train,
        "val": n_val,
        "test": n_test,
    }

    start = 0
    for split, count in split_counts.items():
        dst_dir = os.path.join(output_root, split, class_name)
        os.makedirs(dst_dir, exist_ok=True)
        for file in image_files[start:start + count]:
            src_file = os.path.join(class_path, file)
            dst_file = os.path.join(dst_dir, file)
            shutil.copy2(src_file, dst_file)
        start += count

print("Dataset split and sampling completed.")
