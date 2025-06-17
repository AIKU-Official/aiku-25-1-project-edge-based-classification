import cv2
# import matplotlib.pyplot as plt
# import numpy as np
import os
import shutil
import sys
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

edge_list = ['canny', 'sobel', 'LoG', 'hed', 'rcf']

######### 여기만 바꾸시오 #########
edge_idx = 4  # [0, 4]

input_path = '/home/aikusrv04/hj/dataset/imagenet-1k-subset-v2'  # train, val, test 폴더들이 존재하는 경로

output_path = f'/home/aikusrv04/hj/dataset/imagenet-1k-subset-v2-edge/{edge_list[edge_idx]}'  # train, val, test 폴더들이 들어갈 경로
##################################


### HED ###
if edge_idx == 3:
    deploy_path = '/home/aikusrv04/hj/edge_detection/hed/deploy.prototxt'
    model_path = '/home/aikusrv04/hj/edge_detection/hed/hed_pretrained_bsds.caffemodel'
    net = cv2.dnn.readNetFromCaffe(deploy_path, model_path)
###########

### RCF ###
if edge_idx == 4:
    sys.path.append('/home/aikusrv04/hj/edge_detection')
    from rcf import RCF
    rcf_model = RCF(device=device)
###########

def image_count(input_path, split):
    count = 0

    for code in os.listdir(os.path.join(input_path, split)):
        input_dir = os.path.join(input_path, split, code)

        if not os.path.isdir(input_dir):
            continue
            
        count += sum(os.path.isfile(os.path.join(input_dir, f)) for f in os.listdir(input_dir))

    return count

def canny(input_file, output_file):
    image = cv2.imread(input_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)

    cv2.imwrite(output_file, edges)

def sobel(input_file, output_file):
    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    sobel = cv2.magnitude(sobelx, sobely)
    sobel = cv2.convertScaleAbs(sobel)

    cv2.imwrite(output_file, sobel)

def LoG(input_file, output_file):
    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)

    blurred = cv2.GaussianBlur(image, (3, 3), 0)

    log = cv2.Laplacian(blurred, cv2.CV_64F)
    log_abs = cv2.convertScaleAbs(log)

    _, binary_edge = cv2.threshold(log_abs, 10, 255, cv2.THRESH_BINARY)

    cv2.imwrite(output_file, binary_edge)

def hed(input_file, output_file):
    image = cv2.imread(input_file)
    (H, W) = image.shape[:2]
    image_resized = cv2.resize(image, (500, 500))

    blob = cv2.dnn.blobFromImage(image_resized, scalefactor=1.0, size=(500,500),
                                mean=(104.00698793, 116.66876762, 122.67891434),
                                swapRB=False, crop=False)
    net.setInput(blob)
    hed = net.forward()

    hed = cv2.resize(hed[0, 0], (W, H))
    hed = (255 * hed).astype("uint8")

    cv2.imwrite(output_file, hed)

def rcf(input_file, output_file):
    image = cv2.imread(input_file)

    rcf = rcf_model.detect_edge(image)

    cv2.imwrite(output_file, rcf)

def main():
    print(f'\ncurrent method: {edge_list[edge_idx]}\n')
    
    for split in os.listdir(input_path):
        total_image_count = image_count(input_path, split)
        processed_image_count = 0

        for code in os.listdir(os.path.join(input_path, split)):
            input_dir = os.path.join(input_path, split, code)

            if not os.path.isdir(input_dir):
                continue

            output_dir = os.path.join(output_path, split, code)

            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)

            for file_name in os.listdir(input_dir):
                input_file = os.path.join(input_dir, file_name)

                if not os.path.isfile(input_file):
                    continue

                output_file = os.path.join(output_dir, file_name)

                if edge_idx == 0:
                    canny(input_file, output_file)
                elif edge_idx == 1:
                    sobel(input_file, output_file)
                elif edge_idx == 2:
                    LoG(input_file, output_file)
                elif edge_idx == 3:
                    hed(input_file, output_file)
                elif edge_idx == 4:
                    rcf(input_file, output_file)

                processed_image_count += 1

                if processed_image_count % (total_image_count // 10) == 0:
                    print(f'processed images [{split}]: {processed_image_count}/{total_image_count}')

        print(f'\n{edge_list[edge_idx]} method [{split}]: {image_count(output_path, split)} of {image_count(input_path, split)} was processed.\n')

if __name__=='__main__':
    main()