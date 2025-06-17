import os

DATASET_ROOT_DIR = "./imagenet-1k-subset/val"
SAMPLE_LIST_PATH = "sample_list.txt"

for root, dirs, files in os.walk(DATASET_ROOT_DIR):
    with open(SAMPLE_LIST_PATH, 'a+') as f:
        for file in files:
            f.write(root.split('/')[-1] + " " + file + "\n")
        