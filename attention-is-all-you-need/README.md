# Attention Is All You Need

## 논문 정보

- 제목: Attention Is All You Need
- 저자: Ashish Vaswani 외
- 링크: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

## 구현 상태

구현 중 — 기본 파일 구조를 만들었고, 논문을 읽으며 Transformer의 구성 요소를 하나씩 구현할 예정임.

## 구현할 구성 요소

- Scaled Dot-Product Attention
- Multi-Head Attention
- Positional Encoding
- Position-wise Feed-Forward Network
- Encoder / Decoder
- Transformer

아직 실제 연산은 구현하지 않았다. 각 구성 요소를 구현한 뒤 작은 random tensor로 shape과 forward/backward 동작을 확인할 예정이다.

## 프로젝트 구조

```text
attention-is-all-you-need/
├── README.md
├── src/                       # 모델 구현
├── notebooks/
│   └── train_colab.ipynb      # Colab 학습 및 실험
├── train.py                   # 학습 진입점
└── test.py                    # 로컬 검증
```

## 로컬 테스트

`test.py`에서 synthetic input을 사용해 다음 항목을 확인한다.

- 입력과 출력 shape
- forward pass
- backward pass
- gradient 생성 여부

```bash
python test.py
```

## 학습

학습 코드는 아직 구현하지 않았다. 이후 `train.py`를 학습 진입점으로 사용할 예정이다.

```bash
python train.py
```

## Colab

로컬에서 모델 구조와 기본 동작을 확인한 뒤 GitHub에 푸시한다. 이후 `notebooks/train_colab.ipynb`에서 저장소를 클론 후 데이터셋 준비와 GPU 학습을 진행한다.

## 실험 결과
