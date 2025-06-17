CUDA_VISIBLE_DEVICES=4 python train2.py \
--log-dir ./train_resnet.log \
--data-dir ./dataset/imagenet-1k-subset-v2 \
--save-path ./checkpoint/ImageNet_edges.pth \
# --epochs 30
