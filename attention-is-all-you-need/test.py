"""Local validation entry point for the Attention Is All You Need implementation."""

import torch


def main() -> None:
    torch.manual_seed(0)
    synthetic_input = torch.randn(2, 8, 16, requires_grad=True)

    # TODO: Add shape assertions for the model input and output.
    # TODO: Run and validate a forward pass.
    # TODO: Run a backward pass and validate gradient flow.
    print(f"Synthetic input shape: {tuple(synthetic_input.shape)}")
    print("Transformer-specific validations are not implemented yet.")


if __name__ == "__main__":
    main()
