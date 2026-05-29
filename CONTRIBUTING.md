# Contributing to ExPLoRe

Thank you for your interest!

## Reporting Issues

- **Bug reports**: include the full error traceback, your PyTorch/CUDA version, and the command you ran.
- **Feature requests**: describe the use case and expected behavior.

## Pull Requests

1. Fork the repo and create a feature branch from `main`
2. Make your changes and ensure tests pass: `CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -v`
3. Submit a PR with a clear description of what changed and why

## Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/ExPLoRe.git
cd ExPLoRe
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/ -v
```

All 167 tests should pass. Tests use tiny models (embed_dim=64, depth=2) and run on CPU.

## Code Style

- Follow existing patterns in the codebase
- Keep imports at the top of files
- For MoE-related changes, ensure the dispatch/combine softmax-axis contracts are preserved (tests/test_soft_moe.py)
