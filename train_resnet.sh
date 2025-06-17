CUDA_VISIBLE_DEVICES=4 python train_resnet.py \
--log-dir ./train_resnet.log \
--data-dir ./dataset/imagenet-1k-subset-v2-edge/sobel \
--save-path ./checkpoint/ImageNet-subset-v2-edge-sobel-original-full-finetuned.pth \
# --epochs 30
CUDA_VISIBLE_DEVICES=4 python train_resnet.py \
--log-dir ./train_resnet.log \
--data-dir ./dataset/Dark-ImageNet-subset-v2/naive-0.1 \
--save-path ./checkpoint/ImageNet-subset-v2-edge-sobel-original-full-finetuned-original-fc-finetuning.pth \
--model-path ./checkpoint/ImageNet-subset-v2-edge-sobel-original-full-finetuned.pth
CUDA_VISIBLE_DEVICES=4 python train_resnet.py \
--log-dir ./train_resnet.log \
--data-dir ./dataset/Dark-ImageNet-subset-v2-edge/sobel/naive-0.1 \
--save-path ./checkpoint/ImageNet-subset-v2-edge-sobel-original-full-finetuned-sobel-naive-0.1-fc-finetuning.pth \
--model-path ./checkpoint/ImageNet-subset-v2-edge-sobel-original-full-finetuned.pth
CUDA_VISIBLE_DEVICES=4 python train_resnet.py \
--log-dir ./train_resnet.log \
--data-dir ./dataset/Dark-ImageNet-subset-v2/naive-0.1 \
--save-path ./checkpoint/ImageNet-subset-v2-edge-sobel-original-full-finetuned-original-full-finetuning.pth \
--model-path ./checkpoint/ImageNet-subset-v2-edge-sobel-original-full-finetuned.pth \
--full-finetuning
CUDA_VISIBLE_DEVICES=4 python train_resnet.py \
--log-dir ./train_resnet.log \
--data-dir ./dataset/Dark-ImageNet-subset-v2-edge/sobel/naive-0.1 \
--save-path ./checkpoint/ImageNet-subset-v2-edge-sobel-original-full-finetuned-sobel-naive-0.1-full-finetuning.pth \
--model-path ./checkpoint/ImageNet-subset-v2-edge-sobel-original-full-finetuned.pth \
--full-finetuning

# CUDA_VISIBLE_DEVICES=4 python train_resnet.py \
# --log-dir ./train_resnet.log \
# --data-dir ./dataset/imagenet-1k-subset-v2-edge/canny \
# --save-path ./checkpoint/ImageNet-subset-v2-edge-canny-original-full-pretrained.pth \
# --pretrained
# CUDA_VISIBLE_DEVICES=4 python train_resnet.py \
# --log-dir ./train_resnet.log \
# --data-dir ./dataset/imagenet-1k-subset-v2-edge/hed \
# --save-path ./checkpoint/ImageNet-subset-v2-edge-hed-original-full-pretrained.pth \
# --pretrained
# CUDA_VISIBLE_DEVICES=4 python train_resnet.py \
# --log-dir ./train_resnet.log \
# --data-dir ./dataset/imagenet-1k-subset-v2-edge/LoG \
# --save-path ./checkpoint/ImageNet-subset-v2-edge-LoG-original-full-pretrained.pth \
# --pretrained
# CUDA_VISIBLE_DEVICES=4 python train_resnet.py \
# --log-dir ./train_resnet.log \
# --data-dir ./dataset/imagenet-1k-subset-v2-edge/rcf \
# --save-path ./checkpoint/ImageNet-subset-v2-edge-rcf-original-full-pretrained.pth \
# --pretrained