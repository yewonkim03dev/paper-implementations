"""Training entry point for the SASRec implementation."""

import argparse
from contextlib import AbstractContextManager, nullcontext
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.data import load_sasrec_interactions, split_user_histories
from src.dataset import SASRecTrainDataset, create_sasrec_collate_fn
from src.metrics import RankingMetrics
from src.model import SASRec
from src.training import (
    TrainingResumeState,
    capture_rng_state,
    create_sasrec_optimizer,
    evaluate,
    extract_model_config,
    load_model_checkpoint,
    load_training_checkpoint,
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
    clear_existing: bool = True,
    purge_step: int | None = None,
) -> AbstractContextManager[Any]:
    """TensorBoard SummaryWriter를 컨텍스트 매니저로 열고 신규/재개 실행에 맞게 로그 정리"""
    if not enabled:
        return nullcontext(None)

    from torch.utils.tensorboard import SummaryWriter

    tb_dir = output_dir / "tensorboard"
    if clear_existing and tb_dir.is_dir():
        for stale_event in tb_dir.glob("events.out.tfevents.*"):
            stale_event.unlink()
    tb_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tb_dir), purge_step=purge_step)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """저장 중 중단되어도 불완전한 JSON이 남지 않도록 임시 파일 후 원자적 교체"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _validate_metrics_config(
    metrics_path: Path,
    *,
    model_config: dict[str, Any],
    run_config: dict[str, Any],
) -> dict[str, Any] | None:
    """기존 metrics.json을 읽어 파싱 가능 여부와 현재 모델·실험 설정 일치 여부를 검증"""
    if not metrics_path.exists():
        return None
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("config"), dict):
        return None

    cfg = data["config"]
    expected_pairs: dict[str, Any] = {
        "block_variant": model_config["block_variant"],
        "max_len": model_config["max_len"],
        "hidden_dim": model_config["hidden_dim"],
        "num_blocks": model_config["num_blocks"],
        "num_heads": model_config["num_heads"],
        "dropout": model_config["dropout"],
        "eps": model_config["eps"],
        "padding_mode": run_config["padding_mode"],
        "num_users": run_config["num_users"],
        "num_train_users": run_config["num_train_users"],
        "num_valid_users": run_config["num_valid_users"],
        "num_test_users": run_config["num_test_users"],
        "num_items": run_config["num_items"],
        "batch_size": run_config["batch_size"],
        "lr": run_config["lr"],
        "l2_emb": run_config["l2_emb"],
        "eval_k": run_config["eval_k"],
        "eval_negatives": run_config["eval_negatives"],
        "eval_max_users": run_config["eval_max_users"],
        "eval_seed": run_config["eval_seed"],
        "seed": run_config["seed"],
    }
    mismatches = {
        key: (cfg.get(key), expected)
        for key, expected in expected_pairs.items()
        if cfg.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"기존 metrics.json의 실험 설정이 현재 실행 설정과 일치하지 않음: {mismatches}"
        )
    return data


def _is_completed_metrics(data: dict[str, Any] | None, *, epochs: int) -> bool:
    """검증된 metrics.json이 목표 epoch까지 학습과 test 평가를 정상 완료했는지 확인"""
    if data is None:
        return False
    cfg = data.get("config", {})
    test_block = data.get("test")
    history_rows = data.get("history")
    if cfg.get("epochs") != epochs:
        return False
    if not isinstance(test_block, dict):
        return False
    if not isinstance(test_block.get("hr"), (int, float)) or not math.isfinite(
        float(test_block["hr"])
    ):
        return False
    if not isinstance(test_block.get("ndcg"), (int, float)) or not math.isfinite(
        float(test_block["ndcg"])
    ):
        return False
    if not isinstance(history_rows, list) or len(history_rows) != epochs:
        return False
    return history_rows[-1].get("epoch") == epochs


def _reconcile_best_checkpoint(
    best_checkpoint_path: Path,
    *,
    resume_state: TrainingResumeState,
    model: SASRec,
    optimizer: torch.optim.Optimizer,
    run_config: dict[str, Any],
) -> tuple[int, float, float]:
    """Resume 시 last.pt와 best.pt 간 불일치를 검출·복구해 일관된 Best 상태를 반환"""
    best_epoch = resume_state.best_epoch
    best_valid_hr = resume_state.best_valid_hr
    best_valid_ndcg = resume_state.best_valid_ndcg
    current_model_config = extract_model_config(model)

    if best_checkpoint_path.exists():
        best_ckpt = torch.load(
            best_checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if best_ckpt.get("model_config") not in (None, current_model_config):
            raise ValueError(
                f"{best_checkpoint_path}의 모델 설정이 현재 모델과 일치하지 않음"
            )
        saved_best_run_cfg = best_ckpt.get("run_config")
        if saved_best_run_cfg is not None and saved_best_run_cfg != run_config:
            raise ValueError(
                f"{best_checkpoint_path}의 실험 설정이 현재 실행 설정과 일치하지 않음"
            )

        ckpt_epoch = int(best_ckpt["epoch"])
        ckpt_valid = best_ckpt.get("valid_metrics") or best_ckpt.get(
            "best_valid_metrics"
        )
        if ckpt_valid is None:
            raise ValueError(f"{best_checkpoint_path}에 valid_metrics 정보가 없음")
        ckpt_hr = float(ckpt_valid["hr"])
        ckpt_ndcg = float(ckpt_valid["ndcg"])

        # last.pt 저장 직후 best.pt 갱신 전에 종료된 경우 현재 복원된 모델로 best.pt 복구
        if (
            best_epoch == resume_state.epoch
            and ckpt_epoch != best_epoch
            and resume_state.valid_metrics is not None
        ):
            save_training_checkpoint(
                best_checkpoint_path,
                model,
                optimizer,
                epoch=best_epoch,
                train_loss=resume_state.train_loss,
                valid_metrics=resume_state.valid_metrics,
                best_epoch=best_epoch,
                best_valid_metrics=resume_state.valid_metrics,
                run_config=run_config,
            )
            return best_epoch, best_valid_hr, best_valid_ndcg

        # best.pt가 더 높은 성능을 가졌거나 구버전 last.pt(best_epoch 미기록)인 경우 best.pt 기준으로 동기화
        if (ckpt_ndcg, ckpt_hr) > (best_valid_ndcg, best_valid_hr):
            return ckpt_epoch, ckpt_hr, ckpt_ndcg

        if ckpt_epoch != best_epoch or not (
            math.isclose(ckpt_ndcg, best_valid_ndcg, rel_tol=1e-9, abs_tol=1e-12)
            and math.isclose(ckpt_hr, best_valid_hr, rel_tol=1e-9, abs_tol=1e-12)
        ):
            return ckpt_epoch, ckpt_hr, ckpt_ndcg

        return best_epoch, best_valid_hr, best_valid_ndcg

    # best.pt 파일이 없는 경우
    if best_epoch == 0:
        return 0, -1.0, -1.0
    if best_epoch == resume_state.epoch and resume_state.valid_metrics is not None:
        # 첫 평가 epoch에서 last.pt만 저장되고 best.pt 생성 전 종료된 경우 즉시 복구
        save_training_checkpoint(
            best_checkpoint_path,
            model,
            optimizer,
            epoch=best_epoch,
            train_loss=resume_state.train_loss,
            valid_metrics=resume_state.valid_metrics,
            best_epoch=best_epoch,
            best_valid_metrics=resume_state.valid_metrics,
            run_config=run_config,
        )
        return best_epoch, best_valid_hr, best_valid_ndcg

    raise FileNotFoundError(
        f"last.pt에 best_epoch={best_epoch} 기록이 있으나 체크포인트가 없음: {best_checkpoint_path}"
    )


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="output_dir/last.pt가 있으면 마지막 저장 epoch부터 학습 재개",
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
    metrics_path = output_dir / "metrics.json"
    run_config: dict[str, Any] = {
        "data_name": args.data_path.name,
        "num_users": len(histories),
        "num_train_users": len(dataset),
        "num_valid_users": len(split.valid),
        "num_test_users": len(split.test),
        "num_items": split.num_items,
        "padding_mode": args.padding_mode,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "l2_emb": args.l2_emb,
        "eval_k": args.eval_k,
        "eval_negatives": args.eval_negatives,
        "eval_max_users": args.eval_max_users,
        "eval_seed": eval_seed,
        "seed": args.seed,
    }

    start_epoch = 1
    best_epoch = 0
    best_valid_ndcg = -1.0
    best_valid_hr = -1.0
    history: list[dict[str, Any]] = []
    resumed = False

    if args.resume:
        if not last_checkpoint_path.exists():
            raise FileNotFoundError(
                f"--resume이 지정되었으나 체크포인트 파일이 없음: {last_checkpoint_path}"
            )

        resume_state = load_training_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            expected_run_config=run_config,
            loader_generator=loader_generator,
        )
        if resume_state.epoch > args.epochs:
            raise ValueError(
                f"체크포인트 epoch({resume_state.epoch})이 "
                f"--epochs({args.epochs})보다 큼"
            )

        validated_metrics = _validate_metrics_config(
            metrics_path,
            model_config=extract_model_config(model),
            run_config=run_config,
        )
        best_epoch, best_valid_hr, best_valid_ndcg = _reconcile_best_checkpoint(
            best_checkpoint_path,
            resume_state=resume_state,
            model=model,
            optimizer=optimizer,
            run_config=run_config,
        )
        start_epoch = resume_state.epoch + 1
        history = list(resume_state.history)
        if not history and validated_metrics is not None and isinstance(
            validated_metrics.get("history"), list
        ):
            history = [dict(row) for row in validated_metrics["history"]]
        resumed = True

        if start_epoch > args.epochs and _is_completed_metrics(
            validated_metrics,
            epochs=args.epochs,
        ):
            print(
                f"resumed from {last_checkpoint_path} "
                f"(epoch {resume_state.epoch}/{args.epochs} already completed; skipping)"
            )
            return

        if not resume_state.rng_restored:
            # 구버전 체크포인트(rng_state 미포함) 복원 시 1 epoch와 같은 시퀀스 반복 방지용 fallback
            resume_seed = args.seed + start_epoch - 1
            set_seed(resume_seed)
            loader_generator.manual_seed(resume_seed)

        print(
            f"resumed from {last_checkpoint_path} "
            f"(completed_epoch={resume_state.epoch}, next_epoch={start_epoch}/{args.epochs}, "
            f"best_epoch={best_epoch}, rng_restored={resume_state.rng_restored})"
        )
    else:
        # 신규 학습 시작 시 이전 실험의 잔여 산출물이 섞이지 않도록 정리
        for stale_file in (best_checkpoint_path, last_checkpoint_path, metrics_path):
            stale_file.unlink(missing_ok=True)

    tb_purge_step = (
        start_epoch if (resumed and start_epoch <= args.epochs) else None
    )
    with _open_tensorboard_writer(
        output_dir,
        enabled=not args.no_tensorboard,
        clear_existing=not resumed,
        purge_step=tb_purge_step,
    ) as writer:
        for epoch in range(start_epoch, args.epochs + 1):
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

            current_valid_metrics: RankingMetrics | None = None
            is_better = False
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
                current_valid_metrics = valid_metrics
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
            best_metrics_snapshot = (
                RankingMetrics(
                    hr=best_valid_hr,
                    ndcg=best_valid_ndcg,
                    k=args.eval_k,
                )
                if best_epoch > 0
                else None
            )
            # last.pt를 먼저 갱신한 뒤 best.pt를 저장해 두 파일 간 불일치 시 복원 가능하도록 순서 고정
            save_training_checkpoint(
                last_checkpoint_path,
                model,
                optimizer,
                epoch=epoch,
                train_loss=train_loss,
                valid_metrics=current_valid_metrics,
                best_epoch=best_epoch,
                best_valid_metrics=best_metrics_snapshot,
                history=history,
                run_config=run_config,
                rng_state=capture_rng_state(loader_generator),
            )
            if is_better and current_valid_metrics is not None:
                save_training_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    valid_metrics=current_valid_metrics,
                    best_epoch=best_epoch,
                    best_valid_metrics=current_valid_metrics,
                    run_config=run_config,
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
    _write_json_atomic(metrics_path, summary)


if __name__ == "__main__":
    main()
