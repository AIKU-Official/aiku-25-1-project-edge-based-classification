import os, shutil

SRC_DIR = "./imagenet-1k/val"
DST_DIR = "./imagenet-1k-subset-test/val"
SAMPLE_LIST_FILE = "sample_list.txt"

os.makedirs(DST_DIR, exist_ok=True)

with open(SAMPLE_LIST_FILE, "r") as f:
    sample_list = f.readlines()
    
sample_list = [sample.strip().split(' ') for sample in sample_list]

for class_name, img_name in sample_list:
    src = os.path.join(SRC_DIR, class_name, img_name)
    dst = os.path.join(DST_DIR, class_name, img_name)
    
    os.makedirs(os.path.join(DST_DIR, class_name), exist_ok=True)
    
    shutil.copy2(src, dst)