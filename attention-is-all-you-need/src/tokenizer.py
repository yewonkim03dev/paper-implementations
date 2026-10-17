from collections.abc import Iterable, Iterator, Sequence
from itertools import zip_longest
from pathlib import Path
import unicodedata

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from torch import Tensor

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

PAD_INDEX = 0
UNK_INDEX = 1
BOS_INDEX = 2
EOS_INDEX = 3

SPECIAL_TOKENS = (
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
)


def normalize_text(text: str) -> str:
    # 유니코드 NFC 정규화 후 연속된 공백을 하나의 공백으로 정리한다.
    normalized = unicodedata.normalize("NFC", text)
    return " ".join(normalized.strip().split())


def _iter_preprocessed_pairs(
    pairs: Iterable[tuple[str, str]],
    max_length: int | None = None,
    tokenizer: "BPETokenizer | None" = None,
) -> Iterator[tuple[str, str]]:
    if max_length is not None and max_length <= 0:
        raise ValueError("max_length must be greater than 0")

    for source_text, target_text in pairs:
        source_clean = normalize_text(source_text)
        target_clean = normalize_text(target_text)

        # 빈 문장이 포함된 쌍은 제외한다.
        if not source_clean or not target_clean:
            continue

        if max_length is not None:
            if tokenizer is not None:
                source_len = len(tokenizer.encode_source(source_clean))
                target_len = len(tokenizer.encode_target(target_clean))
            else:
                source_len = len(source_clean.split())
                target_len = len(target_clean.split())

            if source_len > max_length or target_len > max_length:
                continue

        yield source_clean, target_clean


def preprocess_pairs(
    pairs: Iterable[tuple[str, str]],
    max_length: int | None = None,
    tokenizer: "BPETokenizer | None" = None,
) -> list[tuple[str, str]]:
    return list(
        _iter_preprocessed_pairs(
            pairs,
            max_length=max_length,
            tokenizer=tokenizer,
        )
    )


def _iter_aligned_lines(
    source_lines: Iterable[str],
    target_lines: Iterable[str],
) -> Iterator[tuple[str, str]]:
    for line_number, (source_line, target_line) in enumerate(
        zip_longest(source_lines, target_lines),
        start=1,
    ):
        if source_line is None or target_line is None:
            raise ValueError(
                "source and target files must have the same number of lines "
                f"(mismatch at line {line_number})"
            )
        yield source_line, target_line


def load_parallel_corpus(
    source_path: str | Path,
    target_path: str | Path,
    max_length: int | None = None,
    tokenizer: "BPETokenizer | None" = None,
) -> Iterator[tuple[str, str]]:
    source_file_path = Path(source_path)
    target_file_path = Path(target_path)

    with source_file_path.open(encoding="utf-8") as source_file:
        with target_file_path.open(encoding="utf-8") as target_file:
            yield from _iter_preprocessed_pairs(
                _iter_aligned_lines(source_file, target_file),
                max_length=max_length,
                tokenizer=tokenizer,
            )


class BPETokenizer:
    def __init__(self, tokenizer: Tokenizer | None = None):
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN

        self.pad_index = PAD_INDEX
        self.unk_index = UNK_INDEX
        self.bos_index = BOS_INDEX
        self.eos_index = EOS_INDEX

        if tokenizer is None:
            self._tokenizer = self._build_base_tokenizer()
            self._is_trained = False
        else:
            self._tokenizer = tokenizer
            self._verify_special_tokens()
            self._is_trained = True

    @staticmethod
    def _build_base_tokenizer() -> Tokenizer:
        tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
        tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.NFC(),
                normalizers.Strip(),
            ]
        )
        # 공백을 보존하면서 문장부호를 분리해 subword 재사용성을 높인다.
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Metaspace(),
                pre_tokenizers.Punctuation(behavior="isolated"),
            ]
        )
        tokenizer.decoder = decoders.Metaspace()
        return tokenizer

    def _verify_special_tokens(self) -> None:
        expected = {
            PAD_TOKEN: PAD_INDEX,
            UNK_TOKEN: UNK_INDEX,
            BOS_TOKEN: BOS_INDEX,
            EOS_TOKEN: EOS_INDEX,
        }
        for token, expected_index in expected.items():
            actual_index = self._tokenizer.token_to_id(token)
            if actual_index != expected_index:
                raise ValueError(
                    f"Expected {token} to have index {expected_index}, "
                    f"got {actual_index}"
                )

    @property
    def vocab_size(self) -> int:
        if not self._is_trained:
            raise RuntimeError(
                "Tokenizer must be trained or loaded before accessing vocab_size"
            )
        return self._tokenizer.get_vocab_size()

    def train(
        self,
        texts: Iterable[str],
        vocab_size: int = 37000,
        min_frequency: int = 2,
    ) -> None:
        if vocab_size <= len(SPECIAL_TOKENS):
            raise ValueError(
                f"vocab_size must be greater than {len(SPECIAL_TOKENS)}"
            )

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(SPECIAL_TOKENS),
        )
        normalized_texts = (
            normalize_text(text) for text in texts if normalize_text(text)
        )
        self._tokenizer.train_from_iterator(normalized_texts, trainer=trainer)
        self._verify_special_tokens()
        self._is_trained = True

    def train_from_pairs(
        self,
        pairs: Iterable[tuple[str, str]],
        vocab_size: int = 37000,
        min_frequency: int = 2,
    ) -> None:
        # 원논문처럼 source와 target 문장을 함께 사용해 공유 vocabulary를 학습한다.
        def _iter_sentences() -> Iterable[str]:
            for source_text, target_text in pairs:
                if source_clean := normalize_text(source_text):
                    yield source_clean
                if target_clean := normalize_text(target_text):
                    yield target_clean

        self.train(
            _iter_sentences(),
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if not self._is_trained:
            raise RuntimeError(
                "Tokenizer must be trained or loaded before encoding"
            )

        cleaned = normalize_text(text)
        token_ids = (
            list(self._tokenizer.encode(cleaned).ids) if cleaned else []
        )

        if add_bos:
            token_ids = [self.bos_index, *token_ids]
        if add_eos:
            token_ids = [*token_ids, self.eos_index]

        return token_ids

    def encode_source(
        self,
        text: str,
        add_eos: bool = True,
    ) -> list[int]:
        # Encoder 입력 sequence: [w_1, ..., w_n, <eos>]
        return self.encode(text, add_bos=False, add_eos=add_eos)

    def encode_target(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        # Target 전체 sequence: [<bos>, y_1, ..., y_m, <eos>]
        return self.encode(text, add_bos=add_bos, add_eos=add_eos)

    def decode(
        self,
        token_ids: Sequence[int] | Tensor,
        skip_special_tokens: bool = True,
        stop_at_eos: bool = True,
    ) -> str:
        if not self._is_trained:
            raise RuntimeError(
                "Tokenizer must be trained or loaded before decoding"
            )

        if isinstance(token_ids, Tensor):
            if token_ids.dim() != 1:
                raise ValueError("decode expects a 1D tensor of token indices")
            ids = token_ids.tolist()
        else:
            ids = [int(token_id) for token_id in token_ids]

        if stop_at_eos and self.eos_index in ids:
            ids = ids[: ids.index(self.eos_index)]

        if skip_special_tokens:
            special_indices = {
                self.pad_index,
                self.bos_index,
                self.eos_index,
            }
            ids = [
                token_id for token_id in ids if token_id not in special_indices
            ]

        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: str | Path) -> None:
        if not self._is_trained:
            raise RuntimeError(
                "Tokenizer must be trained or loaded before saving"
            )

        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(target_path))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        tokenizer = Tokenizer.from_file(str(path))
        return cls(tokenizer=tokenizer)
