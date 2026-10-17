from collections.abc import Iterator
from itertools import islice
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, get_worker_info

from .tokenizer import BPETokenizer, PAD_INDEX, load_parallel_corpus

TranslationSample = tuple[Tensor, Tensor, Tensor]
TranslationBatch = tuple[Tensor, Tensor, Tensor]


class TranslationDataset(IterableDataset):
    def __init__(
        self,
        source_path: str | Path,
        target_path: str | Path,
        tokenizer: BPETokenizer,
        max_length: int | None = None,
    ):
        super().__init__()

        if max_length is not None and max_length <= 0:
            raise ValueError("max_length must be greater than 0")

        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self) -> Iterator[TranslationSample]:
        pairs = load_parallel_corpus(
            self.source_path,
            self.target_path,
        )

        # DataLoader worker마다 서로 다른 문장 쌍을 처리해 중복을 막는다.
        worker_info = get_worker_info()
        if worker_info is not None:
            pairs = islice(
                pairs,
                worker_info.id,
                None,
                worker_info.num_workers,
            )

        for source_text, target_text in pairs:
            source_ids = self.tokenizer.encode_source(source_text)
            target_ids = self.tokenizer.encode_target(target_text)

            # token 기준 최대 길이를 넘는 문장 쌍은 제외한다.
            if self.max_length is not None:
                if (
                    len(source_ids) > self.max_length
                    or len(target_ids) > self.max_length
                ):
                    continue

            # source: [x_1, ..., x_n, <eos>]
            source = torch.tensor(source_ids, dtype=torch.long)

            # 전체 target: [<bos>, y_1, ..., y_m, <eos>]
            # Decoder 입력: [<bos>, y_1, ..., y_m]
            decoder_input = torch.tensor(target_ids[:-1], dtype=torch.long)

            # 정답: [y_1, ..., y_m, <eos>]
            labels = torch.tensor(target_ids[1:], dtype=torch.long)

            # return (S_source), (S_target), (S_target)
            yield source, decoder_input, labels


def collate_translation_batch(
    batch: list[TranslationSample],
    padding_index: int = PAD_INDEX,
) -> TranslationBatch:
    if not batch:
        raise ValueError("batch must not be empty")

    sources, decoder_inputs, labels = zip(*batch)

    # 각 sequence를 현재 batch의 최대 길이에 맞춰 오른쪽 padding한다.
    # N * (S_source) -> (B, max_S_source)
    source_batch = pad_sequence(
        sources,
        batch_first=True,
        padding_value=padding_index,
    )

    # N * (S_target) -> (B, max_S_target)
    decoder_input_batch = pad_sequence(
        decoder_inputs,
        batch_first=True,
        padding_value=padding_index,
    )
    label_batch = pad_sequence(
        labels,
        batch_first=True,
        padding_value=padding_index,
    )

    # return (B, max_S_source),
    #        (B, max_S_target),
    #        (B, max_S_target)
    return source_batch, decoder_input_batch, label_batch
