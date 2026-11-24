"""Interaction loading, preprocessing, and temporal data splitting."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


UserHistories = dict[int, list[int]]


@dataclass
class InteractionSplit:
    train: UserHistories
    valid: dict[int, int]
    test: dict[int, int]
    num_items: int


def load_sasrec_interactions(path: str | Path) -> UserHistories:
    """저자가 전처리한 ml-1m 파일의 행 순서를 유지한 사용자별 아이템 이력"""
    path = Path(path)
    histories: UserHistories = defaultdict(list)

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            fields = line.split()
            if not fields:
                continue

            try:
                user_id, item_id = map(int, fields) # 두개 만들기
            except ValueError as error:
                raise ValueError(f"잘못된 상호작용 형식 (정수 ID 두 개 필요): {path}:{line_number}") from error

            if user_id < 1 or item_id < 1:
                raise ValueError(f"잘못된 사용자 또는 아이템 ID (1 이상 필요): {path}:{line_number}")

            histories[user_id].append(item_id)

    if not histories:
        raise ValueError(f"상호작용이 없는 파일: {path}")

    return dict(histories)


def split_user_histories(histories: UserHistories) -> InteractionSplit:
    """시간순 사용자 이력의 마지막 두 아이템으로 검증과 테스트 분할"""
    train: UserHistories = {}
    valid: dict[int, int] = {}
    test: dict[int, int] = {}
    num_items = 0

    for user_id, item_ids in histories.items():
        # 임베딩 크기 기준 검증과 테스트를 포함한 최대 아이템 ID
        num_items = max(num_items, max(item_ids, default=0))

        # 기록이 3개 미만인 사용자는 학습에만 포함함 val test 만들 수 없음
        if len(item_ids) < 3:
            train[user_id] = item_ids.copy()
            continue

        train[user_id] = item_ids[:-2]
        valid[user_id] = item_ids[-2]
        test[user_id] = item_ids[-1]

    return InteractionSplit(train=train, valid=valid, test=test, num_items=num_items)
