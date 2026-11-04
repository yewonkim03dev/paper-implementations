import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

from .inference import translate_text
from .tokenizer import BPETokenizer
from .transformer import Transformer


def _tokenize_for_bleu(text: str) -> list[str]:
    # 단어와 문장부호를 분리해 간단한 BLEU token 단위로 사용한다.
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def _ngram_counts(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(tokens[index : index + order])
        for index in range(len(tokens) - order + 1)
    )


def corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[str],
    max_order: int = 4,
    smooth: bool = True,
) -> float:
    if len(hypotheses) != len(references):
        raise ValueError("hypotheses and references must have the same length")
    if not hypotheses:
        raise ValueError("at least one hypothesis is required")
    if max_order <= 0:
        raise ValueError("max_order must be greater than 0")

    matches_by_order = [0] * max_order
    possible_by_order = [0] * max_order
    hypothesis_length = 0
    reference_length = 0

    for hypothesis, reference in zip(hypotheses, references):
        hypothesis_tokens = _tokenize_for_bleu(hypothesis)
        reference_tokens = _tokenize_for_bleu(reference)
        hypothesis_length += len(hypothesis_tokens)
        reference_length += len(reference_tokens)

        for order in range(1, max_order + 1):
            hypothesis_ngrams = _ngram_counts(hypothesis_tokens, order)
            reference_ngrams = _ngram_counts(reference_tokens, order)
            overlap = hypothesis_ngrams & reference_ngrams

            matches_by_order[order - 1] += sum(overlap.values())
            possible_by_order[order - 1] += sum(hypothesis_ngrams.values())

    if hypothesis_length == 0:
        return 0.0

    precisions = []
    smoothing_factor = 1.0
    for matches, possible in zip(matches_by_order, possible_by_order):
        # 짧은 문장에 존재하지 않는 n-gram 차수는 평균에서 제외한다.
        if possible == 0:
            continue
        if matches == 0:
            if not smooth:
                return 0.0
            smoothing_factor *= 2.0
            precisions.append(1.0 / (smoothing_factor * possible))
        else:
            precisions.append(matches / possible)

    if not precisions:
        return 0.0

    geometric_mean = math.exp(
        sum(math.log(precision) for precision in precisions)
        / len(precisions)
    )
    brevity_penalty = min(
        1.0,
        math.exp(1.0 - reference_length / hypothesis_length),
    )
    return 100.0 * brevity_penalty * geometric_mean


def evaluate_bleu(
    model: Transformer,
    pairs: Iterable[tuple[str, str]],
    tokenizer: BPETokenizer,
    max_length: int = 256,
    beam_size: int = 4,
    length_penalty: float = 0.6,
    max_sentences: int | None = None,
) -> float:
    if max_sentences is not None and max_sentences <= 0:
        raise ValueError("max_sentences must be greater than 0")

    hypotheses: list[str] = []
    references: list[str] = []

    for source_text, target_text in pairs:
        hypotheses.append(
            translate_text(
                model,
                tokenizer,
                source_text,
                max_length=max_length,
                beam_size=beam_size,
                length_penalty=length_penalty,
            )
        )
        references.append(target_text)

        if max_sentences is not None and len(hypotheses) >= max_sentences:
            break

    if not hypotheses:
        raise RuntimeError("No sentence pairs were produced for BLEU evaluation")

    return corpus_bleu(hypotheses, references)
