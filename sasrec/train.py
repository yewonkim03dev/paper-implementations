"""Training entry point for the SASRec implementation."""

import argparse
from contextlib import AbstractContextManager, nullcontext
import json
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data import load_sasrec_interactions, split_user_histories
from src.dataset import SASRecTrainDataset, create_sasrec_collate_fn
from src.model import SASRec
from src.training import (
    create_sasrec_optimizer,
    evaluate,
    load_model_checkpoint,
    save_training_checkpoint,
    set_seed,
    train_epoch,
)


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_eval_negatives(raw_value: str) -> int | None:
    normalized = raw_value.strip().lower()
    if normalized in ("all", "none", "full"):
        return None
    try:
        value = int(raw_value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"--eval-negatives는 양의 정수 또는 'all'이어야 함: {raw_value}"
        ) from error
    if value < 1:
        raise argparse.ArgumentTypeError(
            f"--eval-negatives는 1 이상이어야 함: {value}"
        )
    return value


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def _open_tensorboard_writer(
    output_dir: Path,
    *,
    enabled: bool,
) -> AbstractContextManager[Any]:
    """TensorBoard SummaryWriter를 컨텍스트 매니저로 열고 이전 이벤트 파일 섞임을 방지"""
    if not enabled:
        return nullcontext(None)

    from torch.utils.tensorboard import SummaryWriter

    tb_dir = output_dir / "tensorboard"
    if tb_dir.is_dir():
        for stale_event in tb_dir.glob("events.out.tfevents.*"):
            stale_event.unlink()
    tb_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tb_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate SASRec (paper vs author block variants)."
    )

    # 데이터 및 패딩 설정
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/ml-1m.txt"),
    )
    parser.add_argument(
        "--padding-mode",
        choices=("fixed", "dynamic"),
        default="fixed",
    )

    # 모델 하이퍼파라미터 (기본값: ML-1M 논문 설정)
    parser.add_argument(
        "--block-variant",
        choices=("paper", "author"),
        default="author",
    )
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=50)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=1e-8)

    # 학습 하이퍼파라미터
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2-emb", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)

    # 평가 설정
    parser.add_argument("--eval-k", type=int, default=10)
    parser.add_argument(
        "--eval-negatives",
        type=parse_eval_negatives,
        default=100,
        help="평가용 무작위 negative 개수 (예: 100) 또는 전체 아이템 평가 시 'all'",
    )
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-max-users", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)

    # 실행 환경 및 출력 경로
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="미지정 시 outputs/<dataset>/<block-variant>/seed<seed> 디렉터리에 저장",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="배치별 tqdm 진행률 표시 비활성화",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="TensorBoard 이벤트 로그 기록 비활성화",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_len < 1:
        raise ValueError(f"--max-len은 1 이상이어야 함: {args.max_len}")
    if args.hidden_dim < 1:
        raise ValueError(f"--hidden-dim은 1 이상이어야 함: {args.hidden_dim}")
    if args.num_blocks < 1:
        raise ValueError(f"--num-blocks는 1 이상이어야 함: {args.num_blocks}")
    if args.num_heads < 1:
        raise ValueError(f"--num-heads는 1 이상이어야 함: {args.num_heads}")
    if args.hidden_dim % args.num_heads != 0:
        raise ValueError(
            f"--hidden-dim({args.hidden_dim})은 --num-heads({args.num_heads})로 나누어떨어져야 함"
        )
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError(f"--dropout은 [0.0, 1.0) 범위여야 함: {args.dropout}")
    if args.eps <= 0.0:
        raise ValueError(f"--eps는 0보다 커야 함: {args.eps}")
    if args.epochs < 1:
        raise ValueError(f"--epochs는 1 이상이어야 함: {args.epochs}")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size는 1 이상이어야 함: {args.batch_size}")
    if args.num_workers < 0:
        raise ValueError(f"--num-workers는 0 이상이어야 함: {args.num_workers}")
    if args.eval_interval < 1:
        raise ValueError(f"--eval-interval은 1 이상이어야 함: {args.eval_interval}")
    if args.eval_batch_size is not None and args.eval_batch_size < 1:
        raise ValueError(
            f"--eval-batch-size는 1 이상이어야 함: {args.eval_batch_size}"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    device = resolve_device(args.device)
    dataset_name = args.data_path.stem.lower() or "dataset"
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path("outputs") / dataset_name / args.block_variant / f"seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_batch_size = (
        args.eval_batch_size
        if args.eval_batch_size is not None
        else args.batch_size
    )
    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed
    show_progress = not args.no_progress

    # 데이터 로드 및 시간순 leave-one-out 분할
    histories = load_sasrec_interactions(args.data_path)
    split = split_user_histories(histories)
    dataset = SASRecTrainDataset(
        split.train,
        num_items=split.num_items,
        max_len=args.max_len,
    )
    collate_fn = create_sasrec_collate_fn(
        padding_mode=args.padding_mode,
        max_len=args.max_len,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=_seed_worker if args.num_workers > 0 else None,
        generator=loader_generator,
    )

    model = SASRec(
        num_items=split.num_items,
        max_len=args.max_len,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
        block_variant=args.block_variant,
        eps=args.eps,
    ).to(device)
    optimizer = create_sasrec_optimizer(model, lr=args.lr)

    eval_mode_label = (
        "all" if args.eval_negatives is None else str(args.eval_negatives)
    )
    print(
        f"device={device} variant={args.block_variant} "
        f"padding={args.padding_mode} users={len(histories)} "
        f"train_users={len(dataset)} items={split.num_items} "
        f"eval_negatives={eval_mode_label} output_dir={output_dir}"
    )

    best_checkpoint_path = output_dir / "best.pt"
    last_checkpoint_path = output_dir / "last.pt"
    best_epoch = 0
    best_valid_ndcg = -1.0
    best_valid_hr = -1.0
    history: list[dict[str, Any]] = []

    with _open_tensorboard_writer(
        output_dir,
        enabled=not args.no_tensorboard,
    ) as writer:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                l2_emb=args.l2_emb,
                show_progress=show_progress,
            )
            elapsed_sec = time.perf_counter() - epoch_start

            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Time/train_epoch", elapsed_sec, epoch)
                writer.add_scalar(
                    "LearningRate",
                    optimizer.param_groups[0]["lr"],
                    epoch,
                )

            record: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "elapsed_sec": elapsed_sec,
            }

            should_evaluate = (
                epoch % args.eval_interval == 0 or epoch == args.epochs
            )
            if should_evaluate:
                valid_metrics = evaluate(
                    model,
                    split,
                    device,
                    mode="valid",
                    k=args.eval_k,
                    num_negatives=args.eval_negatives,
                    batch_size=eval_batch_size,
                    padding_mode=args.padding_mode,
                    seed=eval_seed,
                    max_users=args.eval_max_users,
                    show_progress=show_progress,
                )
                if writer is not None:
                    writer.add_scalar(
                        f"HR@{args.eval_k}/valid",
                        valid_metrics.hr,
                        epoch,
                    )
                    writer.add_scalar(
                        f"NDCG@{args.eval_k}/valid",
                        valid_metrics.ndcg,
                        epoch,
                    )

                record["valid_hr"] = valid_metrics.hr
                record["valid_ndcg"] = valid_metrics.ndcg

                is_better = (valid_metrics.ndcg > best_valid_ndcg) or (
                    valid_metrics.ndcg == best_valid_ndcg
                    and valid_metrics.hr > best_valid_hr
                )
                if is_better:
                    best_epoch = epoch
                    best_valid_ndcg = valid_metrics.ndcg
                    best_valid_hr = valid_metrics.hr
                    save_training_checkpoint(
                        best_checkpoint_path,
                        model,
                        optimizer,
                        epoch=epoch,
                        train_loss=train_loss,
                        valid_metrics=valid_metrics,
                    )

                print(
                    f"epoch={epoch}/{args.epochs} "
                    f"train_loss={train_loss:.4f} "
                    f"valid_hr@{args.eval_k}={valid_metrics.hr:.4f} "
                    f"valid_ndcg@{args.eval_k}={valid_metrics.ndcg:.4f} "
                    f"time={elapsed_sec:.2f}s"
                )
            else:
                print(
                    f"epoch={epoch}/{args.epochs} "
                    f"train_loss={train_loss:.4f} "
                    f"time={elapsed_sec:.2f}s"
                )

            if writer is not None:
                # Colab Google Drive 등 버퍼/지연 환경에서도 epoch별 TensorBoard 곡선이 바로 갱신되도록 flush
                writer.flush()

            history.append(record)

        save_training_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            epoch=args.epochs,
            train_loss=history[-1]["train_loss"],
        )

        # 가장 높은 valid NDCG@K를 기록한 best.pt로 최종 test 분할 평가
        best_model = load_model_checkpoint(best_checkpoint_path, device=device)
        test_metrics = evaluate(
            best_model,
            split,
            device,
            mode="test",
            k=args.eval_k,
            num_negatives=args.eval_negatives,
            batch_size=eval_batch_size,
            padding_mode=args.padding_mode,
            seed=eval_seed,
            max_users=args.eval_max_users,
            show_progress=show_progress,
        )
        if writer is not None:
            writer.add_scalar(
                f"HR@{args.eval_k}/test",
                test_metrics.hr,
                best_epoch,
            )
            writer.add_scalar(
                f"NDCG@{args.eval_k}/test",
                test_metrics.ndcg,
                best_epoch,
            )
            writer.flush()

        print(
            f"best_epoch={best_epoch} "
            f"valid_hr@{args.eval_k}={best_valid_hr:.4f} "
            f"valid_ndcg@{args.eval_k}={best_valid_ndcg:.4f} "
            f"test_hr@{args.eval_k}={test_metrics.hr:.4f} "
            f"test_ndcg@{args.eval_k}={test_metrics.ndcg:.4f}"
        )

    summary = {
        "config": {
            "data_path": str(args.data_path),
            "output_dir": str(output_dir),
            "block_variant": args.block_variant,
            "padding_mode": args.padding_mode,
            "num_users": len(histories),
            "num_train_users": len(dataset),
            "num_valid_users": len(split.valid),
            "num_test_users": len(split.test),
            "num_items": split.num_items,
            "max_len": args.max_len,
            "hidden_dim": args.hidden_dim,
            "num_blocks": args.num_blocks,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "eps": args.eps,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "l2_emb": args.l2_emb,
            "num_workers": args.num_workers,
            "eval_k": args.eval_k,
            "eval_negatives": args.eval_negatives,
            "eval_batch_size": eval_batch_size,
            "eval_interval": args.eval_interval,
            "eval_max_users": args.eval_max_users,
            "eval_seed": eval_seed,
            "seed": args.seed,
            "device": str(device),
        },
        "best_epoch": best_epoch,
        "best_valid": {
            "hr": best_valid_hr,
            "ndcg": best_valid_ndcg,
            "k": args.eval_k,
        },
        "test": {
            "hr": test_metrics.hr,
            "ndcg": test_metrics.ndcg,
            "k": args.eval_k,
        },
        "history": history,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
