from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim import Adam, Optimizer

from .tokenizer import PAD_INDEX


class LabelSmoothingLoss(nn.Module):
    def __init__(
        self,
        smoothing: float = 0.1,
        padding_index: int = PAD_INDEX,
    ):
        super().__init__()

        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")

        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.padding_index = padding_index

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        # logits: (B, S_target, vocab_size)
        # labels: (B, S_target)
        if logits.dim() != 3:
            raise ValueError("logits must have shape (B, S_target, vocab_size)")
        if labels.shape != logits.shape[:2]:
            raise ValueError("labels must match the first two logits dimensions")

        vocab_size = logits.size(-1)
        if vocab_size <= 2:
            raise ValueError("vocab_size must be greater than 2")
        if not 0 <= self.padding_index < vocab_size:
            raise ValueError("padding_index must be inside the vocabulary")

        # (B, S_target, vocab_size)
        log_probabilities = F.log_softmax(logits, dim=-1)
        non_padding = labels != self.padding_index

        # batch 전체가 padding이어도 gradient가 연결된 0을 반환한다.
        if not torch.any(non_padding):
            return logits.sum() * 0.0

        with torch.no_grad():
            # 정답 token과 padding token을 제외한 나머지 token에
            # smoothing 확률을 동일하게 분배한다.
            smoothing_value = self.smoothing / (vocab_size - 2)
            target_distribution = torch.full_like(
                log_probabilities,
                smoothing_value,
            )
            target_distribution.scatter_(
                dim=-1,
                index=labels.unsqueeze(-1),
                value=self.confidence,
            )
            target_distribution[..., self.padding_index] = 0.0
            target_distribution.masked_fill_(
                ~non_padding.unsqueeze(-1),
                0.0,
            )

        # token별 cross entropy: (B, S_target)
        token_losses = -(target_distribution * log_probabilities).sum(dim=-1)

        # padding을 제외한 token 수로 평균낸 scalar loss
        return token_losses.sum() / non_padding.sum()


class TransformerLRScheduler:
    def __init__(
        self,
        optimizer: Optimizer,
        d_model: int,
        warmup_steps: int = 4000,
        factor: float = 1.0,
    ):
        if d_model <= 0:
            raise ValueError("d_model must be greater than 0")
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be greater than 0")
        if factor <= 0:
            raise ValueError("factor must be greater than 0")

        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_number = 0

    def rate(self, step: int | None = None) -> float:
        current_step = self.step_number if step is None else step
        if current_step <= 0:
            return 0.0

        # d_model^(-0.5) * min(step^(-0.5), step * warmup^(-1.5))
        return self.factor * (self.d_model ** -0.5) * min(
            current_step ** -0.5,
            current_step * (self.warmup_steps ** -1.5),
        )

    def step(self) -> float:
        # 원논문 schedule의 현재 step learning rate를 optimizer에 적용한다.
        self.step_number += 1
        learning_rate = self.rate()
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        return learning_rate

    def state_dict(self) -> dict[str, int]:
        return {"step_number": self.step_number}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        step_number = state_dict["step_number"]
        if step_number < 0:
            raise ValueError("step_number must not be negative")

        self.step_number = step_number
        learning_rate = self.rate()
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = learning_rate


def create_transformer_optimizer(
    parameters: Iterable[Tensor],
    d_model: int,
    warmup_steps: int = 4000,
    factor: float = 1.0,
) -> tuple[Adam, TransformerLRScheduler]:
    # 원논문 Adam 설정: beta_1=0.9, beta_2=0.98, epsilon=1e-9
    optimizer = Adam(
        parameters,
        lr=0.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = TransformerLRScheduler(
        optimizer,
        d_model=d_model,
        warmup_steps=warmup_steps,
        factor=factor,
    )
    return optimizer, scheduler
