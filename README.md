# 프로젝트명 

📢 2025년 1학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다. 

## 소개

![image](https://github.com/user-attachments/assets/0a5992e5-6933-4e5c-b2c7-164ef1581a41)

오늘의 포켓몬은 뭘까~~요~?

이미지 전체를 주고 분류를 시키는 건 이제 너무 쉽죠. 물체 실루엣만 주고 분류를 시킨다면 잘할 수 있을까요? 물체 edge만 준다면요?

... 하지만 기초컴퓨터비전 수업 프로젝트로 하기에는 너무나도 가벼운 주제였습니다. B0를 맞기는 싫어 좀 더 학술적인 상황을 가정해보았습니다. 

![image](https://github.com/user-attachments/assets/64aa119e-189a-4e39-8511-8ce25daa5486)

기존 이미지 분류 모델들은 ImageNet과 같은 데이터셋에서 학습되었는데요, 그 중 대부분의 이미지들은 밝은 상황에서 촬영된 이미지입니다. 그렇기에 어두운 상황에서 촬영된 이미지에 대한 분류 성능은 저하되는 모습을 보입니다. 

어두운 환경에서는 물체의 색깔이나 질감 등, 디테일한 정보가 손실됩니다. 물체의 윤곽선이나 edge 정보를 활용하여, 정보 손실을 보완할 수는 없을까요?

이를 확인하기 위해, 물체의 edge 및 실루엣을 검출하고 이를 바탕으로 image classification task를 수행해보았습니다. 

## 방법론

- Baseline: 기존 분류 모델을 어두운 이미지로 finetuning
- Edge: 어두운 이미지로부터 edge detection을 수행하여 edge 정보를 추출
- Silhouette: 어두운 이미지에 segmentation을 수행하여 object mask를 추출
- 목표: Edge와 silhouette 정보를 잘 활용하여, baseline보다 좋은 성능으로 어두운 이미지 classfication을 해보자!

### Silouette-Based
- 이미지에서 추출한 object mask를 기반으로 분류를 진행

### Edge-Based
- 이미지에서 추출한 edge를 기반으로 분류를 진행

### Dual Channel
- 어두운 이미지 + 이미지에서 추출한 edge를 모두 활용하여 분류를 진행

## 환경 설정

```
git clone edge-based-classification
cd edge-based-classification

conda create -n edge-based-classification python=3.8.18
conda activate edge-based-classification

pip install -r requirements.txt
```

## 사용 방법

### 기존 이미지(IamgeNet)를 변형하여 어두운 이미지 데이터셋 생성
```
cd dataset
sh darkenize-v2.sh
```

### 필요한 폴더 및 딥러닝 기반 엣지 검출 모델 사용을 위한 weight 추가
- `checkpoint` 폴더 추가
- `edge_detection/hed/`에 `deploy.prototxt`, `hed_pretrained_bsds.caffemodel` 추가 [[Google Drive](https://drive.google.com/drive/folders/1nMgMYNcLuW8O8O7Uu2raZl0d6lthE347)]
- `edge_detection/rcf/weights/`에 `only-final-lr-0.01-iter-130000.pth` 추가

### Edge detection 수행
```
cd edge_detection
python edge_detect.py
```

### 모델 학습 및 평가
```
sh train_resnet.sh
eval_resnet.sh
```

### Dual method 학습 및 평가
```
dual_train.sh
dual_eval.sh
```

## 예시 결과

### 기존 이미지
<img src="https://github.com/user-attachments/assets/736d89fc-f2c2-46c5-981c-4e0f372a6160" width="400"/>

### 어둡게 변형한 이미지
<img src="https://github.com/user-attachments/assets/f5741e31-60c5-477b-8543-18a4a1ac3df1" width="400"/>

### Edge 추출
<img src="https://github.com/user-attachments/assets/22989ec3-9948-48a1-9ab9-fa67e79b3bfc" width="400"/>
<img src="https://github.com/user-attachments/assets/af14cf56-048f-4515-a05d-cae101577b6b" width="400"/>

### 성능 정리

| Method | Top-5 Accuracy |
|-------|-------|
| Baseline | **65.22**  |
| Silouette-Based  | 23.59 |
| Edge-Based | 30.07 |
| Dual Channel | **57.48** |
| Dual Channel + Edge from original image | **67.85** |

- 어두운 이미지에서의 edge 추출 성능은 일반 이미지에서 추출할 때보다 저하되는 결과
- 더 나은 edge detection을 수행할 수 있는 상황을 가정하여, 일반 이미지 또는 덜 어둡게 변형한 이미지로부터 edge를 추출하여 실험을 진행
- 일반 이미지에서 추출한 edge를 활용할 경우, baseline보다 높은 분류 성능 -> edge detection이 성능 bottleneck이라고 볼 수 있음

## 팀원

- [문정민](https://github.com/strn18): 문제 정의, 엣지 검출, 모델 학습 및 평가
- [이현진](https://github.com/hyunjin09): 모델 학습 및 평가
