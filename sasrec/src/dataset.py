"""Sequence dataset, positive/negative item sampling, and left-padding helpers."""

from collections.abc import Callable, Sequence
from functools import partial
import random
from typing import Literal

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .data import UserHistories


PaddingMode = Literal["fixed", "dynamic"]
SASRecSample = tuple[Tensor, Tensor, Tensor]
SASRecBatch = tuple[Tensor, Tensor, Tensor]


def pad_sequences_left(
    sequences: Sequence[Sequence[int] | Tensor],
    *,
    padding_mode: PaddingMode = "dynamic",
    max_len: int | None = None,
    padding_value: int = 0,
    device: torch.device | str | None = None,
) -> Tensor:
    """시퀀스 목록을 고정 길이(max_len) 또는 배치 최대 길이(dynamic)에 맞춰 왼쪽 패딩"""
    if not sequences:
        raise ValueError("패딩할 시퀀스 목록이 비어 있음")
    if padding_mode not in ("fixed", "dynamic"):
        raise ValueError(f"알 수 없는 padding_mode: {padding_mode}")

    lengths = [len(seq) for seq in sequences]
    if min(lengths) < 1:
        raise ValueError("빈 시퀀스는 패딩할 수 없음")
    batch_max_len = max(lengths)

    if max_len is not None:
        if max_len < 1:
            raise ValueError(f"잘못된 최대 시퀀스 길이: {max_len}")
        if batch_max_len > max_len:
            raise ValueError(
                f"시퀀스 길이가 max_len({max_len})을 초과함: {batch_max_len}"
            )

    if padding_mode == "fixed":
        if max_len is None:
            raise ValueError("padding_mode='fixed'일 때는 max_len이 필요함")
        target_len = max_len
    else:
        target_len = batch_max_len

    # 0으로 초기화된 텐서를 먼저 만듬, 시퀀스 길이 * target_len
    # ex) batch size=4, max_len=10 -> 4 * 10 크기의 0 텐서
    padded = torch.full(
        (len(sequences), target_len),
        padding_value,
        dtype=torch.long,
        device=device,
    )
    for row_idx, seq in enumerate(sequences):
        # 이미 텐서 형태일 경우 device만 바꾸고, 아니면 하나 새로 만듬
        seq_tensor = (
            seq.to(device=device, dtype=torch.long)
            if isinstance(seq, Tensor)
            else torch.tensor(seq, dtype=torch.long, device=device)
        )
        if seq_tensor.ndim != 1:
            raise ValueError(f"개별 시퀀스는 1차원이어야 함: {tuple(seq_tensor.shape)}")
        
        # 0으로 채워진 텐서[row_idx행, 목표길이 - 시퀀스 길이:] -> padded에 대입
        # 기존 시퀀스 텐서에 0을 채우는게 아니라 미리 만들어진 0텐서에 기존 시퀀스 값들을 인덱스에 맞게 대입
        padded[row_idx, target_len - seq_tensor.numel() :] = seq_tensor

    return padded


def collate_sasrec_batch(
    batch: Sequence[SASRecSample],
    *,
    padding_mode: PaddingMode = "dynamic",
    max_len: int | None = None,
) -> SASRecBatch:
    """패딩 전 샘플 리스트를 받아 옵션(fixed / dynamic)에 맞춰 왼쪽 패딩된 배치 생성"""
    if not batch:
        raise ValueError("배치가 비어 있음")

    inputs, positives, negatives = zip(*batch)
    return (
        pad_sequences_left(inputs, padding_mode=padding_mode, max_len=max_len),
        pad_sequences_left(positives, padding_mode=padding_mode, max_len=max_len),
        pad_sequences_left(negatives, padding_mode=padding_mode, max_len=max_len),
    )


# DataLoader는 collate_fn 호출 시 batch만 전달하므로
# partial을 사용해 padding_mode, max_len을 미리 지정
# 실제 함수 실행은 DataLoader가 batch를 전달할 때 이루어짐
# lambda 대신 partial을 사용하여 멀티워커 환경에서 직렬화 지원

# 학습 샘플 목록(Sequence[SASRecSample])을 입력받아
# 패딩된 배치(SASRecBatch)를 반환하는 호출 가능 객체 생성
def create_sasrec_collate_fn(
    *,
    padding_mode: PaddingMode = "fixed",
    max_len: int | None = None,
) -> Callable[[Sequence[SASRecSample]], SASRecBatch]:
    """DataLoader에 전달할 왼쪽 패딩 collate_fn 생성 (멀티워커 직렬화 호환)"""
    if padding_mode not in ("fixed", "dynamic"):
        raise ValueError(f"알 수 없는 padding_mode: {padding_mode}")
    if max_len is not None and max_len < 1:
        raise ValueError(f"잘못된 최대 시퀀스 길이: {max_len}")
    if padding_mode == "fixed" and max_len is None:
        raise ValueError("padding_mode='fixed'일 때는 max_len이 필요함")

    return partial(
        collate_sasrec_batch,
        padding_mode=padding_mode,
        max_len=max_len,
    )


class SASRecTrainDataset(Dataset[SASRecSample]):
    """사용자별 학습 이력에서 패딩 전 (입력, positive, negative) 시퀀스 생성"""

    def __init__(
        self,
        train_histories: UserHistories,
        num_items: int,
        max_len: int = 200,
    ) -> None:
        if num_items < 1:
            raise ValueError(f"잘못된 아이템 수: {num_items}")
        if max_len < 1:
            raise ValueError(f"잘못된 최대 시퀀스 길이: {max_len}")

        self.num_items = num_items
        self.max_len = max_len
        self.histories: list[tuple[list[int], set[int]]] = []

        for user_id, item_ids in train_histories.items():
            if len(item_ids) < 2:
                continue

            seen_items = set(item_ids)
            if min(seen_items) < 1 or max(seen_items) > num_items:
                raise ValueError(f"학습 이력에 범위를 벗어난 아이템 ID: 사용자 {user_id}")
            if len(seen_items) == num_items:
                continue

            """
            학습 가능한 사용자당 인덱스 하나
            학습 이력 [1, 2, 3, 8, 12], max_len=2
            최근 3개 [3, 8, 12] 보관
            negative 제외 목록은 전체 이력 {1, 2, 3, 8, 12}
            """
            self.histories.append((item_ids[-(max_len + 1):], seen_items))

        if not self.histories:
            raise ValueError("학습 샘플을 만들 수 있는 사용자 이력 없음")

    def __len__(self) -> int:
        return len(self.histories)

    # __getitem__ 호출마다 Negative를 새로 샘플링
    # Epoch마다 다양한 미관측 아이템을 학습할 수 있음
    def __getitem__(self, index: int) -> SASRecSample:
        item_ids, seen_items = self.histories[index]

        # 입력보다 한 칸 뒤의 다음 아이템
        input_ids = item_ids[:-1]
        positive_ids = item_ids[1:]

        # 샘플 조회마다 유효 위치별 미관측 아이템 하나씩 균등 추출
        negative_ids = []
        for _ in positive_ids:
            negative_id = random.randint(1, self.num_items)
            while negative_id in seen_items:
                negative_id = random.randint(1, self.num_items)
            negative_ids.append(negative_id)

        """
        학습 이력 [3, 8, 12], max_len=4, num_items=14
        반환 (패딩 전): 입력 [3, 8], positive [8, 12], negative [6, 14] (추출 예시)
        왼쪽 패딩(fixed=max_len 또는 dynamic=배치 최대 길이)은 collate_fn에서 수행
        """
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(positive_ids, dtype=torch.long),
            torch.tensor(negative_ids, dtype=torch.long),
        )
