# Paper Implementations

A personal research and development portfolio for implementing the core models from deep learning papers in PyTorch and validating them through experiments.

Paper reviews and study notes are maintained separately on my blog. This repository focuses on implementation, testing, training, and reproducible experiment code across multiple papers.

## Implementations

| Paper | Status |
| --- | --- |
| Attention Is All You Need | In Progress |

## Workflow

```text
Local implementation
  -> Synthetic input validation
  -> Forward / backward validation
  -> Unit tests
  -> GitHub
  -> Google Colab training and experiments
```

Model architecture and lightweight tests are developed locally. GPU training, dataset preparation, and larger experiments are performed in Google Colab after cloning the repository.

## Repository Structure

```text
paper-implementations/
├── .venv/                         # Shared local environment (not tracked)
├── requirements.txt               # Common dependencies
├── templates/                     # Reusable project scaffold
└── attention-is-all-you-need/     # First paper implementation
```

Each paper directory owns its source code, local test entry point, training entry point, Colab notebook, and implementation documentation.

## Development Environment

Create and activate the shared environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Project-specific dependencies can be added when an implementation requires them.
