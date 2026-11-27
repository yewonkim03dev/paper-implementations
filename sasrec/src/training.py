"""Training and evaluation helpers."""

from collections.abc import Iterable
from pathlib import Path
import random
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Adam, Optimizer

from .data import InteractionSplit
from .dataset import PaddingMode, SASRecBatch, pad_sequences_left
from .metrics import RankingMetrics, compute_target_ranks, hit_rate_at_k, ndcg_at_k
from .model import SASRec


def set_seed(seed: int) -> None:
    """재현 가능한 학습과 평가 >> Python/PyTorch 난수 시드 고정"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_sasrec_optimizer(
    model: SASRec,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.98),
) -> Adam:
    """SASRec 기본 Adam 옵티마이저 생성 (저자 코드 기준 lr=1e-3, beta2=0.98)"""
    if lr <= 0.0:
        raise ValueError(f"lr은 0보다 커야 함: {lr}")
    return Adam(model.parameters(), lr=lr, betas=betas)


def compute_bce_loss(
    pos_scores: Tensor,
    neg_scores: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """패딩 위치를 제외한 유효 위치에 대해 positive/negative BCE loss 계산"""
    if pos_scores.shape != neg_scores.shape:
        raise ValueError(
            f"pos_scores와 neg_scores의 shape이 일치해야 함: "
            f"{tuple(pos_scores.shape)} vs {tuple(neg_scores.shape)}"
        )
    if valid_mask.shape != pos_scores.shape:
        raise ValueError(
            f"valid_mask의 shape이 score와 일치해야 함: "
            f"{tuple(valid_mask.shape)} vs {tuple(pos_scores.shape)}"
        )
    if valid_mask.dtype != torch.bool:
        raise ValueError(f"valid_mask는 bool 텐서여야 함: {valid_mask.dtype}")

    # 유효한 학습 데이터가 없으면 손실을 0으로 반환
    # 계산그래프 유지 목적 
    if not torch.any(valid_mask):
        return (pos_scores.sum() + neg_scores.sum()) * 0.0

    valid_pos = pos_scores[valid_mask]
    valid_neg = neg_scores[valid_mask]

    pos_loss = F.binary_cross_entropy_with_logits(
        valid_pos,
        torch.ones_like(valid_pos),
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        valid_neg,
        torch.zeros_like(valid_neg),
    )
    return pos_loss + neg_loss


def train_epoch(
    model: SASRec,
    batches: Iterable[SASRecBatch],
    optimizer: Optimizer,
    device: torch.device | str,
    *,
    l2_emb: float = 0.0,
) -> float:
    """한 epoch 동안 SASRec을 학습하고 유효 위치 기준 평균 loss 반환"""
    if l2_emb < 0.0:
        raise ValueError(f"l2_emb는 0 이상이어야 함: {l2_emb}")

    # 학습 모드
    model.train()
    total_loss = 0.0
    total_positions = 0
    steps = 0

    for input_ids, positive_ids, negative_ids in batches:
        # (B, S), (B, S), (B, S)
        input_ids = input_ids.to(device)
        positive_ids = positive_ids.to(device)
        negative_ids = negative_ids.to(device)
        
        # 유효한 학습 위치 개수 계산
        valid_mask = positive_ids != 0
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            continue
        
        # 이전 배치 계산 기울기 초기화
        optimizer.zero_grad(set_to_none=True)

        # (B, S, D)
        features = model(input_ids)
        # (B, S), (B, S) -> (B, S, 2)
        # 마지막 차원을 새로 생성해서 결합 = 사용자별, 시퀀스 위치별로 Positive와 Negative 아이템을 묶음
        candidates = torch.stack((positive_ids, negative_ids), dim=-1)
        scores = model.score_items(features, candidates)

        # 나머지 차원 유지
        loss = compute_bce_loss(scores[..., 0], scores[..., 1], valid_mask)
        if l2_emb > 0.0: # 정규화 강도
            # 0번 패딩 임베딩 제외한 모든 아이템 임베딩 제곱합 * 정규화 강도
            loss = loss + l2_emb * model.item_embedding.weight[1:].pow(2).sum()

        loss.backward()
        optimizer.step()

        steps += 1

        # 유효 위치 기준 가중 평균
        # SASRec에서는 사용자마다 시퀀스 길이가 다름 = 패딩 개수도 다름
        # 각 배치에 있는 패딩의 개수가 다르기 때문에 실제 학습에 사용된 위치 개수도 다름(모든 배치의 전체 텐서 크기가 동일하더라도 패딩을 제거하고 나면 실제 손실 계산에 사용되는 위치의 개수가 달라짐)
        # BCE Loss 계산시에 패딩 위치는 제거된 텐서가 반환됨(valid_mask를 인덱스로 사용)
        # loss 자체가 이미 그 배치에서 평균을 낸 값 => 전체 데이터 평균 손실을 구하려면 <배치별 평균> 을 손실 총합으로 복원
        # => valid_count(평균을 계산할 때 나눴던 데이터 개수)를 다시 곱해서 원래 손실 총합을 복원
        total_loss += loss.item() * valid_count
        total_positions += valid_count

    if steps == 0:
        raise RuntimeError("유효한 학습 배치가 생성되지 않음")

    return total_loss / total_positions # 전체 유효 위치 개수로 나눔


def _sample_unseen_negatives(
    num_items: int,
    seen_items: set[int],
    num_negatives: int,
    rng: random.Random,
) -> list[int]:
    """관측 이력에 없는 중복 없는 negative 아이템을 최대 num_negatives개 추출"""
    num_unseen = num_items - len(seen_items)
    if num_unseen <= 0:
        return []

    sample_size = min(num_negatives, num_unseen)
    # 미관측 아이템 비율이 충분하면 rejection sampling, 좁으면 전체 미관측 목록에서 추출
    if len(seen_items) + sample_size < num_items // 2:
        sampled: set[int] = set()
        negatives: list[int] = []
        while len(negatives) < sample_size:
            candidate = rng.randint(1, num_items)
            if candidate not in seen_items and candidate not in sampled:
                sampled.add(candidate)
                negatives.append(candidate)
        return negatives

    unseen_pool = [
        item_id
        for item_id in range(1, num_items + 1)
        if item_id not in seen_items
    ]
    return rng.sample(unseen_pool, sample_size)


def evaluate(
    model: SASRec,
    split: InteractionSplit,
    device: torch.device | str,
    *,
    mode: Literal["valid", "test"] = "valid",
    k: int = 10,
    num_negatives: int | None = 100,
    batch_size: int = 128,
    padding_mode: PaddingMode = "dynamic",
    seed: int = 0,
    max_users: int | None = None,
) -> RankingMetrics:
    """validation 또는 test 분할에 대해 HR@K와 NDCG@K 계산
       Leave-One-Out >> 사용자별 마지막 아이템은 테스트, 직전 아이템은 검증, 나머지는 학습에 사용
          Validation >> 학습 이력 -> 검증 아이템 예측
          Test >> 학습 이력 + 검증 아이템 -> 테스트 아이템 예측

    - mode="valid": 입력 train[u], 정답 valid[u]
    - mode="test": 입력 train[u] + [valid[u]], 정답 test[u]
    - num_negatives=100: 정답 1개 + 무작위 미관측 negative 100개 (논문 방식)
    - num_negatives=None: 전체 미관측 아이템 대상 Full Ranking
    - padding_mode="dynamic" | "fixed": 배치 최대 길이 패딩 또는 max_len 고정 패딩
    """
    if mode not in ("valid", "test"):
        raise ValueError(f"알 수 없는 평가 모드: {mode}")
    if k < 1:
        raise ValueError(f"k는 1 이상이어야 함: {k}")
    if num_negatives is not None and num_negatives < 1:
        raise ValueError(f"num_negatives는 1 이상 또는 None이어야 함: {num_negatives}")
    if batch_size < 1:
        raise ValueError(f"batch_size는 1 이상이어야 함: {batch_size}")
    if padding_mode not in ("fixed", "dynamic"):
        raise ValueError(f"알 수 없는 padding_mode: {padding_mode}")
    if max_users is not None and max_users < 1:
        raise ValueError(f"max_users는 1 이상 또는 None이어야 함: {max_users}")
    if split.num_items < 1:
        raise ValueError(f"잘못된 아이템 수: {split.num_items}")

    # 평가 모드에 따라서 사용자별 정답 아이템 저장(valid, test)
    targets_by_user = split.valid if mode == "valid" else split.test
    user_ids = sorted(targets_by_user)
    if not user_ids:
        raise ValueError(f"평가할 사용자가 없는 분할: {mode}")

    # 평가 전용 로컬 RNG를 사용해 학습 RNG와 분리하고 모델 비교 시 동일 후보군 보장
    rng = random.Random(seed)
    if max_users is not None and len(user_ids) > max_users:
        user_ids = rng.sample(user_ids, max_users)

    max_len = model.position_embedding.num_embeddings # model 최대 시퀀스 길이
    eval_entries: list[tuple[list[int], int, set[int]]] = [] # 입력 아이템 시퀀스, 정답 아이템 id, 이미 소비한 아이템 집합
    
    for user_id in user_ids:
        train_history = split.train.get(user_id, [])
        if mode == "valid":
            history = train_history # 학습 이력만 사용해서 검증 아이템 예측
        else:
            valid_item = split.valid.get(user_id)
            # 학습 이력 + 검증 아이템 모두 사용 -> 테스트 아이템 예측
            history = [*train_history, valid_item] if valid_item is not None else train_history

        if not history:
            continue

        target_item = targets_by_user[user_id]
        # negative 샘플링에서 제외할 기존 소비 + 정답 아이템 합집합 생성
        seen_items = set(history) | {target_item}
        # 미관측 아이템이 하나도 없으면 비교 대상 negative가 없으므로 제외
        if len(seen_items) >= split.num_items:
            continue
        
        # 최근 소비 이력이 중요 -> 최대 시퀀스 길이 넘어가는 과거 이력은 제거
        # 최근 max_len개의 소비 이력, 정답 아이템, 기존 소비 아이템 집합 >> 평가 데이터로 저장
        eval_entries.append((history[-max_len:], target_item, seen_items))

    if not eval_entries:
        raise RuntimeError(f"평가 가능한 사용자가 없음: {mode}")

    model.eval()
    all_ranks: list[Tensor] = []

    with torch.no_grad():
        all_item_ids = (
            torch.arange(1, split.num_items + 1, device=device, dtype=torch.long)
            if num_negatives is None
            else None # Sampled Ranking
        )

        for start in range(0, len(eval_entries), batch_size):
            batch_entries = eval_entries[start : start + batch_size]
            sequences = [entry[0] for entry in batch_entries]

            # 왼쪽 패딩 처리
            input_ids = pad_sequences_left(
                sequences,
                padding_mode=padding_mode,
                max_len=max_len,
                device=device,
            )

            # 왼쪽 패딩 시퀀스의 마지막 위치가 다음 아이템 예측 시점: (B, D)
            last_features = model(input_ids)[:, -1, :]
            
            if num_negatives is not None: # Sampled Ranking으로 호출된 경우
                candidate_rows: list[list[int]] = []
                for _, target_item, seen_items in batch_entries:

                    # 소비 안한 negatives를 Sampled Ranking으로 추출
                    negatives = _sample_unseen_negatives(
                        num_items=split.num_items,
                        seen_items=seen_items,
                        num_negatives=num_negatives,
                        rng=rng,
                    )
                    # 미관측 아이템이 num_negatives보다 적으면 0으로 채우고 점수를 -inf 마스킹
                    pad_count = num_negatives - len(negatives)
                    candidate_rows.append([target_item, *negatives, *([0] * pad_count)])

                candidates = torch.tensor(candidate_rows, dtype=torch.long, device=device)
                scores = model.score_items(last_features, candidates)
                if (candidates == 0).any(): 
                    # 패딩 아이템의 점수를 음의 무한대로 변경 -> 순위에 영향을 주지 못하게
                    scores = scores.masked_fill(candidates == 0, float("-inf"))
                # HR@K, NDCG@K 계산
                batch_ranks = compute_target_ranks(scores, target_indices=0)

            else: # Full Raning으로 호출된 경우
                assert all_item_ids is not None
                # last_features(B, D), all_item_ids(num_items,) 
                # (B, num_items): 열 인덱스 item_id - 1이 실제 item_id에 대응
                scores = model.score_items(last_features, all_item_ids)

                # 사용자별 정답 아이템 ID를 scores의 열 인덱스로 변환
                # 아이템 ID는 1-based 텐서 인덱스는 0-based이므로 1을 차감
                target_indices = torch.tensor(
                    [entry[1] - 1 for entry in batch_entries],
                    dtype=torch.long,
                    device=device,
                )
                for row_idx, (_, target_item, seen_items) in enumerate(batch_entries):
                    # 정답 아이템 자체는 재방문 이력이 있더라도 마스킹하지 않음
                    # 정답 아이템은 마스킹하지 않는 이유는 seen_items에는 기존 소비 아이템뿐만 아니라 정답 아이템도 포함됨
                    # if item_id != target_item 조건이 없다면 -> 정답 아이템의 점수까지 -inf됨
                    masked_indices = [
                        item_id - 1
                        for item_id in seen_items
                        if item_id != target_item
                    ]
                    if masked_indices:
                        scores[row_idx, masked_indices] = float("-inf")
                batch_ranks = compute_target_ranks(scores, target_indices=target_indices)

            all_ranks.append(batch_ranks.cpu())

    ranks = torch.cat(all_ranks, dim=0)
    return RankingMetrics(
        hr=hit_rate_at_k(ranks, k=k),
        ndcg=ndcg_at_k(ranks, k=k),
        k=k,
    )


def extract_model_config(model: SASRec) -> dict[str, Any]:
    """체크포인트 복원을 위해 SASRec 생성자 설정 추출"""
    num_blocks = len(model.blocks)
    num_heads = model.blocks[0].attention.num_heads if num_blocks > 0 else 1
    eps = model.blocks[0].attention_norm.eps if num_blocks > 0 else model.norm.eps
    return {
        "num_items": model.item_embedding.num_embeddings - 1,
        "max_len": model.position_embedding.num_embeddings,
        "hidden_dim": model.item_embedding.embedding_dim,
        "num_blocks": num_blocks,
        "num_heads": num_heads,
        "dropout": model.embedding_dropout.p,
        "block_variant": model.block_variant,
        "eps": eps,
        "final_norm_eps": model.norm.eps,
    }


def save_training_checkpoint(
    path: str | Path,
    model: SASRec,
    optimizer: Optimizer,
    *,
    epoch: int,
    train_loss: float,
    valid_metrics: RankingMetrics | None = None,
) -> None:
    """모델 설정, 가중치, 옵티마이저 상태와 검증 지표를 체크포인트로 저장"""
    if epoch < 1:
        raise ValueError(f"epoch은 1 이상이어야 함: {epoch}")

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_metrics": (
                {
                    "hr": valid_metrics.hr,
                    "ndcg": valid_metrics.ndcg,
                    "k": valid_metrics.k,
                }
                if valid_metrics is not None
                else None
            ),
            "model_config": extract_model_config(model),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )


def load_model_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> SASRec:
    """저장된 체크포인트에서 SASRec 모델을 복원해 eval 모드로 반환"""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    model = SASRec(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model
