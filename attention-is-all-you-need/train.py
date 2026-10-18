"""Training entry point for the Attention Is All You Need implementation."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data import TranslationDataset, collate_translation_batch
from src.tokenizer import BPETokenizer, load_parallel_corpus
from src.training import (
    LabelSmoothingLoss,
    create_transformer_optimizer,
    save_training_checkpoint,
    train_epoch,
)
from src.transformer import Transformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--train-target", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("artifacts/tokenizer.json"),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
    )

    parser.add_argument("--vocab-size", type=int, default=37000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--learning-rate-factor", type=float, default=1.0)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_tokenizer(args: argparse.Namespace) -> BPETokenizer:
    if args.tokenizer_path.exists():
        return BPETokenizer.load(args.tokenizer_path)

    tokenizer = BPETokenizer()
    training_pairs = load_parallel_corpus(
        args.train_source,
        args.train_target,
    )
    tokenizer.train_from_pairs(
        training_pairs,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    tokenizer.save(args.tokenizer_path)
    return tokenizer


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    tokenizer = prepare_tokenizer(args)

    dataset = TranslationDataset(
        args.train_source,
        args.train_target,
        tokenizer,
        max_length=args.max_length,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_translation_batch,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_len=args.max_length,
        padding_index=tokenizer.pad_index,
    ).to(device)

    loss_function = LabelSmoothingLoss(
        smoothing=args.label_smoothing,
        padding_index=tokenizer.pad_index,
    )
    optimizer, scheduler = create_transformer_optimizer(
        model.parameters(),
        d_model=args.d_model,
        warmup_steps=args.warmup_steps,
        factor=args.learning_rate_factor,
    )

    print(f"device={device} vocab_size={tokenizer.vocab_size}")
    for epoch in range(1, args.epochs + 1):
        average_loss = train_epoch(
            model,
            data_loader,
            loss_function,
            optimizer,
            scheduler,
            device,
            log_interval=args.log_interval,
        )
        print(f"epoch={epoch} loss={average_loss:.4f}")

        checkpoint_path = args.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        save_training_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            epoch=epoch,
            loss=average_loss,
        )


if __name__ == "__main__":
    main()
