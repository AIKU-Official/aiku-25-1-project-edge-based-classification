# python darkenizer.py --method naive --src-dir ./imagenet-1k-subset-v2 --dst-dir ./Dark-ImageNet-subset-v2
python darkenizer.py --method naive --dark-opt 0.1 --src-dir ./imagenet-1k-subset-v2 --dst-dir ./Dark-ImageNet-subset-v2
python darkenizer.py --method HSV --src-dir ./imagenet-1k-subset-v2 --dst-dir ./Dark-ImageNet-subset-v2
python darkenizer.py --method HSV --dark-opt 0.1 --src-dir ./imagenet-1k-subset-v2 --dst-dir ./Dark-ImageNet-subset-v2
python darkenizer.py --method gamma --src-dir ./imagenet-1k-subset-v2 --dst-dir ./Dark-ImageNet-subset-v2
python darkenizer.py --method gamma --dark-opt 10 --src-dir ./imagenet-1k-subset-v2 --dst-dir ./Dark-ImageNet-subset-v2
