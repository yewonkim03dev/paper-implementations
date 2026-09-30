# Paper Implementations

딥러닝 논문을 읽고 핵심 모델을 PyTorch로 직접 구현해보는 저장소입니다.

논문을 읽으며 정리한 내용은 블로그에 따로 기록합니다. 이 저장소에는 모델 코드와 테스트, 학습 및 실험에 필요한 파일만 모을 예정입니다.

## Implementations

| 논문                      | 상태    |
| ------------------------- | ------- |
| Attention Is All You Need | 구현 중 |

## Workflow

```text
논문 읽기
  -> 로컬에서 모델 구현
  -> Synthetic input으로 shape 확인
  -> Forward / backward 검증
  -> 필요한 테스트 작성
  -> GitHub
  -> Google Colab에서 학습 및 실험
```

로컬에서는 모델 구조를 구현하고 작은 입력으로 동작을 확인합니다. GPU가 필요한 학습과 실제 데이터셋을 사용한 실험은 리포지토리를 코랩에 클론한 이후에 진행합니다.

## Repository Structure

```text
paper-implementations/
├── .venv/                         # 로컬 공용 가상환경 (Git 제외)
├── requirements.txt               # 공통 dependency
├── templates/                     # 새 구현을 시작할 때 복사할 기본 구조
└── attention-is-all-you-need/     # 첫 번째 논문 구현
```

각 논문 디렉터리에는 모델 코드와 로컬 테스트, 학습 코드, Colab notebook, README를 함께 둡니다. 새로운 논문을 구현할 때는 `templates/`를 복사해서 시작합니다.

```bash
cp -R templates project-name
```

복사한 뒤 프로젝트 이름과 README, Colab notebook의 `PROJECT_NAME`을 수정하면 됩니다.

## Development Environment

가상환경은 저장소 루트의 `.venv` 하나를 공용으로 사용합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

현재 `requirements.txt`에는 여러 구현에서 공통으로 사용할 dependency만 넣어두었습니다. 특정 논문에서 추가 패키지가 필요해지면 구현하면서 추가할 예정입니다.
