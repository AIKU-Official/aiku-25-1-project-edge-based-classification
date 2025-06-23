# 프로젝트명 

📢 2025년 1학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다

## 소개
 
(프로젝트를 소개해주세요)

## 방법론

(문제를 정의하고 이를 해결한 방법을 가독성 있게 설명해주세요)

## 환경 설정

```
git clone mamama
cd mamama

conda create -n mamama python=3.8.18
conda activate mamama

pip install -r requirements.txt
```

## 사용 방법

1. 필요한 폴더 및 딥러닝 기반 엣지 검출 모델 사용을 위한 weight 추가
- `checkpoint` 폴더 추가
- `edge_detection/hed/`에 `deploy.prototxt`, `hed_pretrained_bsds.caffemodel` 추가 [[Google Drive](https://drive.google.com/drive/folders/1nMgMYNcLuW8O8O7Uu2raZl0d6lthE347)]
- `edge_detection/rcf/weights/`에 `only-final-lr-0.01-iter-130000.pth` 추가

(프로젝트 실행 방법 (명령어 등)을 적어주세요.)

## 예시 결과

(사용 방법을 실행했을 때 나타나는 결과나 시각화 이미지를 보여주세요)

## 팀원

(프로젝트에 참여한 팀원의 이름과 깃헙 프로필 링크, 역할을 작성해주세요)

- [문정민](https://github.com/strn18): 문제 정의, 모델 학습 및 평가
- [이현진](https://github.com/hyunjin09): 모델 학습 및 평가
