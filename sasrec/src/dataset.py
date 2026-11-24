"""Sequence dataset and positive/negative item sampling."""

import random

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .data import UserHistories


class SASRecTrainDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """사용자별 학습 이력에서 입력, positive, negative 생성"""

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

            # 최근 max_len + 1개와 negative에서 제외할 전체 학습 이력
            self.histories.append((item_ids[-(max_len + 1):], seen_items))

        if not self.histories:
            raise ValueError("학습 샘플을 만들 수 있는 사용자 이력 없음")

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        item_ids, seen_items = self.histories[index]

        # 입력보다 한 칸 뒤의 다음 아이템
        input_ids = item_ids[:-1]
        positive_ids = item_ids[1:]

        # 유효 위치마다 미관측 아이템 하나를 균등 추출
        negative_ids = []
        for _ in positive_ids:
            negative_id = random.randint(1, self.num_items)
            while negative_id in seen_items:
                negative_id = random.randint(1, self.num_items)
            negative_ids.append(negative_id)

        # 세 텐서 모두 (S,) = (max_len,)
        padding = [0] * (self.max_len - len(input_ids))
        return (
            torch.tensor(padding + input_ids, dtype=torch.long),
            torch.tensor(padding + positive_ids, dtype=torch.long),
            torch.tensor(padding + negative_ids, dtype=torch.long),
        )
