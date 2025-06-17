mkdir imagenet-1k
cd imagenet-1k

# validation set 다운로드 
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar 

# validation set 압축해제
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash


# train set 다운로드
cd /home/dataset/imagenet-1k

wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

#train set 압축해제

mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar
rm -f ILSVRC2012_img_train.tar (만약 원본 압축파일을 지우려면)
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..