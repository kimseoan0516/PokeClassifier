# PokeClassifier

PyTorch 전이 학습 기반 포켓몬 이미지 분류기입니다. 포켓몬 이미지를 업로드하면 150종 중 어떤 포켓몬인지 예측하고, 한글 이름과 타입 정보와 함께 Top-5 결과를 보여줍니다.

## 데모

<img src="https://github.com/user-attachments/assets/b1b2ecdc-9e39-4833-b8f6-880d622ec204" />

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/6f92ec1f-508d-4c6e-b3d5-81e19a0e732a" width="450"/></td>
    <td><img src="https://github.com/user-attachments/assets/07f84b3e-eb0f-4d91-8bd9-32ed8b1ea640" width="450"/></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/07ff0059-c043-41e6-948c-87b225df25de" width="450"/></td>
    <td><img src="https://github.com/user-attachments/assets/769506eb-a507-4358-9a12-3b0d871e0df0" width="450"/></td>
  </tr>
</table>

<img src="https://github.com/user-attachments/assets/41bfe01f-d0a1-4fc8-93d4-90e2e917e437" />

## 데이터셋

[7,000 Labeled Pokemon](https://www.kaggle.com/datasets/lantian773030/pokemonclassification) — 150종, 약 7,000장 (Kaggle)

데이터 분할: **학습 70% / 검증 15% / 테스트 15%**

## 실험 결과

**백본 모델**, **사전학습 가중치 사용 여부**, **파인튜닝 범위**의 영향을 비교하는 4가지 실험을 진행했습니다.

| # | 백본 | 사전학습 | 파인튜닝 범위 | Test Acc | Precision | Recall | F1 |
|---|------|----------|---------------|----------|-----------|--------|----|
| Exp 1 | ResNet18 | ✗ | 전체 (scratch) | 28.3% | 28.5% | 28.7% | 24.2% |
| Exp 2 | ResNet18 | ✓ | Head만 (frozen) | 66.9% | 69.1% | 67.9% | 65.2% |
| Exp 3 | ResNet50 | ✓ | 부분 (layer3+4+head) | **96.1%** | **96.5%** | **96.5%** | **96.1%** |
| Exp 4 | EfficientNet-B0 | ✓ | 전체 파인튜닝 | 85.1% | 84.0% | 84.6% | 82.1% |

### 주요 결과 분석

- **사전학습 가중치의 효과**: Exp 1 (28.3%) → Exp 2 (66.9%), 추가 추론 비용 없이 +38.6%p 향상
- **더 큰 백본이 유리**: ResNet50 부분 파인튜닝(96.1%)이 EfficientNet-B0 전체 파인튜닝(85.1%)보다 우수 — 이 데이터셋에서는 파인튜닝 범위보다 모델 용량이 더 중요
- **부분 파인튜닝이 전체 파인튜닝을 이길 수 있음**: 초기 레이어를 고정하면 정규화 효과와 함께 ImageNet 특징을 보존하여 가장 좋은 성능을 달성

## 학습 곡선

<table>
  <tr>
    <td><img src="results/learning_curves.png" alt="손실 곡선"/></td>
    <td><img src="results/accuracy_curves.png" alt="정확도 곡선"/></td>
  </tr>
</table>

![실험 비교](results/comparison.png)

## 설치

```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 전체 실험 학습

```bash
python run_experiments.py --data_dir path/to/PokemonData
```

특정 실험만 학습하려면:

```bash
python train.py --data_dir path/to/PokemonData --exp exp3_resnet50_partial
```

### 2. 평가 및 그래프 생성

```bash
python evaluate.py --data_dir path/to/PokemonData
```

### 3. 데모 실행

```bash
streamlit run demo.py
```

브라우저에서 `http://localhost:8501` 접속

## 프로젝트 구조

```
PokeClassifier/
├── config.py            # 4개 실험 설정
├── dataset.py           # 데이터 로드, 증강, 분할
├── models.py            # ResNet18 / ResNet50 / EfficientNet-B0
├── train.py             # 학습 루프
├── evaluate.py          # 평가 지표, 혼동 행렬, 학습 곡선 생성
├── demo.py              # Streamlit 데모 앱
├── pokemon_data.py      # 포켓몬 한글 이름 및 타입 데이터
├── run_experiments.py   # 전체 실험 일괄 실행
├── kaggle_train.py      # Kaggle 노트북용 단일 파일 버전 (GPU)
├── requirements.txt
└── results/             # 학습 결과 저장 (학습 후 생성)
    ├── learning_curves.png
    ├── accuracy_curves.png
    ├── comparison.png
    └── exp{1-4}_*/
        ├── results.json
        ├── classification_report.txt
        └── confusion_matrix.png
```

## 파인튜닝 전략

| 모드 | 설명 |
|------|------|
| `scratch` | 랜덤 초기화로 전체 레이어 학습 |
| `frozen` | 백본 고정, 분류 헤드만 학습 |
| `partial` | 초기 레이어(layer1/2) 고정, layer3/4 + 헤드 학습 |
| `full` | 사전학습 가중치에서 전체 레이어 파인튜닝 |
