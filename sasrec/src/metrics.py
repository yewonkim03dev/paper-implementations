"""Top-K recommendation metrics."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RankingMetrics:
    hr: float
    ndcg: float
    k: int


def _validate_ranks(ranks: Tensor, k: int) -> None:
    if k < 1:
        raise ValueError(f"k는 1 이상이어야 함: {k}")
    if ranks.ndim != 1 or ranks.numel() == 0:
        raise ValueError(f"ranks는 비어 있지 않은 1차원 텐서여야 함: {tuple(ranks.shape)}")
    if ranks.is_floating_point() or ranks.dtype == torch.bool:
        raise ValueError(f"ranks는 정수형 텐서여야 함: {ranks.dtype}")
    if (ranks < 0).any():
        raise ValueError("ranks에는 0 이상의 값만 포함되어야 함")


def compute_target_ranks(
    scores: Tensor,
    target_indices: int | Tensor = 0,
) -> Tensor:
    """후보 점수 (B, C)에서 각 샘플의 정답 아이템 0-based 순위 (B,) 계산"""
    if scores.ndim != 2 or scores.size(0) == 0 or scores.size(1) == 0:
        raise ValueError(f"scores는 비어 있지 않은 2차원 (B, C) 텐서여야 함: {tuple(scores.shape)}")
    if torch.isnan(scores).any():
        raise ValueError("scores에 NaN이 포함됨")

    batch_size, num_candidates = scores.shape

    if isinstance(target_indices, int):
        if isinstance(target_indices, bool) or target_indices < 0 or target_indices >= num_candidates:
            raise ValueError(f"잘못된 타깃 인덱스 (0 이상 {num_candidates} 미만 필요): {target_indices}")
        
        # 정답 아이템 인덱스 생성
        index_tensor = torch.full(
            (batch_size, 1),
            target_indices,
            device=scores.device,
            dtype=torch.long,
        )
    else:
        if target_indices.ndim != 1 or target_indices.size(0) != batch_size:
            raise ValueError(
                f"target_indices는 길이 {batch_size}의 1차원 텐서여야 함: {tuple(target_indices.shape)}"
            )
        if target_indices.is_floating_point() or target_indices.dtype == torch.bool:
            raise ValueError(f"target_indices는 정수형 텐서여야 함: {target_indices.dtype}")
        if (target_indices < 0).any() or (target_indices >= num_candidates).any():
            raise ValueError(f"target_indices가 범위를 벗어남 (0 이상 {num_candidates} 미만 필요)")

        index_tensor = target_indices.to(device=scores.device, dtype=torch.long).unsqueeze(1)
    
    # 정답 아이템 점수 추출(열방향 따라가면서)
    target_scores = scores.gather(dim=1, index=index_tensor)
    if not torch.isfinite(target_scores).all():
        raise ValueError("타깃 아이템 점수는 유한한 값이어야 함")

    # 동점 후보도 정답보다 앞서거나 같은 것으로 보수적 처리(상수 출력 모델의 HR=1.0 방지)
    # 정답 아이템보다 점수가 높은 아이템의 개수를 세면 정답 아이템의 순위를 알 수 있음
    # >=를 사용하고 1을 빼면 동점인 후보가 모두 앞에 있다고 가정
    # 타깃 자신은 항상 scores >= target_scores를 만족하므로 1을 빼서 0-based 순위로 맞춤
    return (scores >= target_scores).sum(dim=1) - 1


def hit_rate_at_k(ranks: Tensor, k: int = 10) -> float:
    """0-based 순위 텐서 (B,)에서 HR@K 계산"""
    _validate_ranks(ranks, k)
    return (ranks < k).to(torch.float64).mean().item()


def ndcg_at_k(ranks: Tensor, k: int = 10) -> float:
    """단일 정답(사용자 한 명당 평가에 사용하는 실제 정답 아이템이 하나) 기준 0-based 순위 텐서 (B,)에서 NDCG@K 계산"""
    _validate_ranks(ranks, k)
    hits = ranks < k # k안에 들었는지, 0-based라서 <

    # 순위에 따른 할인 점수 계산
    discounts = 1.0 / torch.log2(ranks.to(torch.float64) + 2.0) # 2를 더하는 이유는 0-based 고려(로그 밑이 2)

    # 조건(hits)에 따라서 discount, 0.0 중 하나를 선택하고 평균내서 파이썬 숫자로 반환
    # 상위 k에 포함되지 않은 경우(조건 hits) 0.0 부여
    return torch.where(hits, discounts, 0.0).mean().item()


def compute_ranking_metrics(
    scores: Tensor,
    k: int = 10,
    target_indices: int | Tensor = 0,
) -> RankingMetrics:
    """후보 점수 (B, C)에서 HR@K와 NDCG@K를 함께 계산"""
    ranks = compute_target_ranks(scores, target_indices=target_indices)
    return RankingMetrics(
        hr=hit_rate_at_k(ranks, k=k),
        ndcg=ndcg_at_k(ranks, k=k),
        k=k,
    )
