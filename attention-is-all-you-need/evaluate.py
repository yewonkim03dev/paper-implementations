"""Evaluate a trained Transformer checkpoint with corpus BLEU."""

import argparse
from pathlib import Path

from src.device import resolve_device
from src.evaluation import evaluate_bleu
from src.inference import load_transformer_checkpoint
from src.tokenizer import BPETokenizer, load_parallel_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=0.6)
    parser.add_argument("--max-sentences", type=int)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    tokenizer = BPETokenizer.load(args.tokenizer_path)
    model = load_transformer_checkpoint(args.checkpoint, device=device)
    max_length = (
        args.max_length
        if args.max_length is not None
        else int(model.config["max_len"])
    )

    pairs = load_parallel_corpus(
        args.source,
        args.target,
        max_length=max_length,
        tokenizer=tokenizer,
    )
    bleu = evaluate_bleu(
        model,
        pairs,
        tokenizer,
        max_length=max_length,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        max_sentences=args.max_sentences,
    )
    print(f"BLEU={bleu:.2f}")


if __name__ == "__main__":
    main()
