# Attention Is All You Need

## 논문 정보

- 제목: Attention Is All You Need
- 저자: Ashish Vaswani 외
- 링크: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

## 구현 상태

모델 구조와 학습에 필요한 기본 코드를 구현했다. 로컬에서는 작은 tensor와 문장으로 동작을 확인했고, 실제 데이터 학습은 Colab에서 진행할 예정이다.

## 구현할 구성 요소

- Scaled Dot-Product Attention
- Multi-Head Attention
- Positional Encoding
- Position-wise Feed-Forward Network
- Encoder / Decoder
- Transformer
- BPE tokenizer와 병렬 코퍼스 스트리밍
- 학습 및 검증 루프
- Greedy / Beam Search decoding
- BLEU 평가

## 프로젝트 구조

```text
attention-is-all-you-need/
├── README.md
├── src/                       # 모델, 데이터, 학습 및 추론 코드
├── notebooks/
│   └── train_colab.ipynb      # Colab 학습 및 실험
├── train.py                   # 학습 진입점
├── translate.py               # 문장 번역
├── evaluate.py                # BLEU 평가
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

source와 target은 같은 줄끼리 번역 쌍을 이루는 텍스트 파일을 사용한다. validation 파일은 생략할 수 있다.

```bash
python train.py \
  --train-source data/train.en \
  --train-target data/train.de \
  --validation-source data/valid.en \
  --validation-target data/valid.de
```

토크나이저가 없으면 학습 데이터로 새로 만든다. checkpoint는 epoch별로 저장하고, validation loss가 가장 낮은 모델은 `best.pt`에 저장한다.

## 번역 및 평가

```bash
python translate.py "A man is walking." \
  --checkpoint checkpoints/best.pt \
  --tokenizer-path artifacts/tokenizer.json

python evaluate.py \
  --source data/test.en \
  --target data/test.de \
  --checkpoint checkpoints/best.pt \
  --tokenizer-path artifacts/tokenizer.json
```

`evaluate.py`의 BLEU는 로컬 확인용이다. 논문 결과와 직접 비교할 때는 논문과 같은 tokenization 및 BLEU 설정을 맞춰야 한다.

## Colab

`notebooks/train_colab.ipynb`를 Colab에서 열고 위에서부터 실행하면 된다. 노트북은 구현 브랜치를 가져온 뒤 Multi30K 영어-독일어 데이터를 내려받고, tokenizer와 checkpoint를 Google Drive에 저장한다. 학습이 끝나면 예시 문장 번역과 test set 일부의 BLEU 평가까지 이어서 실행한다.

기본값은 전체 흐름을 확인하기 위한 축소 모델이다. 원 논문의 WMT14 학습 결과를 그대로 재현하는 설정은 아니다.

## 실험 결과

아직 실제 데이터 학습 전이다.
