CUDA_VISIBLE_DEVICES=4 python dual_train_resnet.py \
--log-dir ./dual_train.log \
--data-dir ./dataset/Dark-ImageNet-subset-v2/naive-0.1 \
--edge-dir ./dataset/imagenet-1k-subset-v2-edge/sobel \
--save-path ./checkpoint/Dark-ImageNet-subset-v2-dual-sobel-original-edge-naive-0.1-img-unfreeze.pth \
--pretrained \
--epochs 10