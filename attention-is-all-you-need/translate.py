"""Translate one sentence with a trained Transformer checkpoint."""

import argparse
from pathlib import Path

from src.device import resolve_device
from src.inference import load_transformer_checkpoint, translate_text
from src.tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--length-penalty", type=float, default=0.6)
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

    translated = translate_text(
        model,
        tokenizer,
        args.text,
        max_length=max_length,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
    )
    print(translated)


if __name__ == "__main__":
    main()
