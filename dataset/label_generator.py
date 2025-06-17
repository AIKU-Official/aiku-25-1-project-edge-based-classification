import json
import urllib.request

output_dir = "./labels"

# ================= Label
url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
labels_path = 'imagenet_classes.txt'
urllib.request.urlretrieve(url, labels_path)

# Load into a Python list
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

with open(f"{output_dir}/imagenet_class_labels.json", "w") as f:
    json.dump(labels, f, indent=2)

# ================= synset/class mapping
url = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
with urllib.request.urlopen(url) as response:
    imagenet_map = json.load(response)

# Save to local file
with open(f"{output_dir}/imagenet_class_index.json", "w") as f:
    json.dump(imagenet_map, f, indent=2)

# Convert to: synset → index
synset_to_idx = {v[0]: int(k) for k, v in imagenet_map.items()}

# Save to local file
with open(f"{output_dir}/synset_to_idx.json", "w") as f:
    json.dump(synset_to_idx, f, indent=2)