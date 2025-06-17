CUDA_VISIBLE_DEVICES=5 python dual_eval_resnet.py \
--log-dir ./dual_eval.log \
--data-dir ./dataset/Dark-ImageNet-subset-v2/naive-0.1 \
--edge-dir ./dataset/imagenet-1k-subset-v2-edge/sobel \
--model-path ./checkpoint/Dark-ImageNet-subset-v2-dual-sobel-original-edge-naive-0.1-img-unfreeze.pth \
--pretrained 