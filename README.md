# 2022 인공지능 온라인 경진대회

## KoreanBeef팀 소고기 분류 문제

### 최종 스코어

| public       | Private     | Final         |
| ------------ | ----------- | ------------- |
| 0.9483128554 | 0.945578642 | 0.94639890602 |

### 코드 구조

```
${KoreanBeef}
├── config
│   ├── predict_config.yaml
│   ├── preprocess_config.yaml
│   └── train_config.yaml
├── data/
│   ├── train.zip
│   ├── test.zip
│   └── sample_submission.csv
├── models
│   ├── convnext.py
│   ├── customnet.py
│   ├── ...
│   └── utils.py
├── modules
│   ├── datasets.py
│   ├── earlystoppers.py
│   ├── losses.py
│   ├── metrics.py
│   ├── optimizers.py
│   ├── recorders.py
│   ├── trainer.py
│   └── utils.py
├── polygon.ipynb
├── predict.py
├── preprocess.py
├── train.py
├── README.md
├── Preprocess.ipynb
├── detectron2.ipynb
├── ensemble.ipynb
└── train_predict.ipynb
```

- config: 학습/추론에 필요한 파라미터 등을 기록하는 yaml 파일
- data: 학습/추론에 필요한 소고기 이미지 데이터 폴더
- models:
  - utils.py: config에서 지정한 모델 클래스를 불러와 리턴하는 파일
- modules:
  - datasets.py: dataset 클래스
  - earlystoppers.py: loss가 지정된 에폭 수 이상 개선되지 않을 경우 학습을 멈추는 early stopper 클래스
  - losses.py: config에서 지정한 loss function을 리턴
  - metrics.py: config에서 지정한 metric을 리턴
  - optimizers.py: config에서 지정한 optimizer를 리턴
  - recorders.py: 로그와 learnig curve 등을 기록
  - trainer.py: 에폭 별로 수행할 학습 과정
  - utils.py: 여러 확장자 파일을 불러오거나 여러 확장자로 저장하는 등의 함수가 포함된 파일
- preprocess.py or Preprocess.ipynb: 학습/추론에 필요한 데이터 로드 작업
- train.py or train_predict.ipynb: 학습
- predict or train_predict.ipynb: 추론
- polygon.ipynb: Detectron2의 image segmentation model을 돌리기 위한 전처리
- detectron2.ipynb: Detectron2 학습 및 결과
- ensemble.ipynb: 모델 블랜딩

---

### Preprocess

**목록에서 Preprocess를 한 번 실행 시키고 이후에는 실행시킬 필요 없음.**

1. 데이터 폴더 생성/준비
   1. 아래 구조와 같이 데이터 폴더를 생성하고 train.zip, test.zip, sample_submission.csv 파일들을 데이터 디렉터리로 이동

```
${data}
├── train.zip
├── test.zip
└── sample_submission.csv
```

2. 'config/preprocess_config.yaml' 수정

   1. DIRECTORY/dataset: train.zip, test.zip, sample_submission.csv 파일들이 위치한 폴더의 경로 지정

3. 'python preprocess.py' 실행
   1. 위 스크립트를 실행하면 아래와 같이 데이터가 준비 됨

```
${DATA}
├── 00_source/
│   ├── images/
│   │   ├── 'aaaaaa.jpg'
│   │   ├── 'bbbbbb.jpg'
│   │   └──  ...
│   └── grade_labels.csv
├── 01_splitdataset/
│   ├── train/
│   │   ├── images/
│   │   │   ├── 'aaaaaa.jpg'
│   │   │   ├── 'bbbbbb.jpg'
│   │   └── grade_labels.csv
│   ├── val/
│   │   ├── images/
│   │   │   ├── 'aaaaaa.jpg'
│   │   └── grade_labels.csv
│   └── test/
│       ├── images/
│       │   ├── 'abcd.jpg'
│       └── test_images.csv
├── sample_submission.csv
├── train.zip
└── test.zip
```

    2. 폴더 구조
        - 00_source: 전체 학습(원천) 데이터 폴더 ('trian.zip'에 들어있는 파일들)
            - images: 모든 학습 데이터 이미지 파일들
            - grade_labels.csv: 모든 학습 데이터 이미지명과 라벨(등급)이 있는 csv
        - 01_splitdataset: 학습 데이터를 임의로 Train/Validation으로 나눈 결과와 Test 데이터 폴더
            - train: 원천 학습 데이터 중 일부를 샘플링하여 만든 Train 데이터
                - images: Train 데이터 이미지 파일들
                - grade_labels.csv: Train 데이터 이미지명과 라벨(등급)이 있는 csv
            - val: 원천 학습 데이터 중 일부를 샘플링하여 만든 Validation 데이터
                - images: Validation 데이터 이미지 파일들
                - grade_labels.csv: Validation 데이터 이미지명과 라벨(등급)이 있는 csv
            - test: Test 데이터 ('test.zip'에 들어있는 파일들)
                - images: Test 데이터 이미지 파일들
                - test_images.csv: Test 데이터 이미지명이 있는 csv
        - sample_submission.csv: 제출 파일 예시 csv

### 학습

1. 'config/train_config.yaml' 수정
   1. DIRECTORY/dataset: 01_splitdataset 폴더의 경로 지정
   2. TRAINER/model: 수정
   3. MODEL 안에 모델 넣어줌
2. 'train.py 또는 train_predict.ipynb' 실행
3. 'results/train/' 내에 결과 폴더가 저장됨

### 추론

1. 'config/predict_config.yaml' 수정
   1. DIRECTORY/dataset: 01_splitdataset 폴더의 경로 지정
   2. DIRECTORY/sample: sample_subission.csv 파일의 경로 지정
   3. TRAIN/train_serial: 파라미터를 불러올 train serial number (result/train 내 폴더명) 지정
2. 'predict or train_predict.ipynb' 실행
3. 'results/predict/' 내에 결과 폴터가 생기고 (predictions.csv)이 저장됨

### 앙상블

1. 각 모델별 추론 결과 csv 6개 가져오기
2. 'ensemble.ipynb' 실행
   - 총 6개 결과 중에 4개 이상은 몰아주기 / 4개 미만은 평균값에 소수점 제하고 +1
3. 지정한 경로에 최종 결과 csv 저장됨.
