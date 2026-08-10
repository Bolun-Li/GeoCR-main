# Contributing to GeoCR

Thank you for helping improve GeoCR. Bug reports, documentation fixes, reproducibility notes, and focused code changes
are welcome.

## Before opening an issue

- Search existing issues first.
- Use one of the issue templates when possible.
- For bugs, include the operating system, Python/PyTorch/CUDA versions, dataset configuration, exact command, and the
  smallest relevant log excerpt.
- Do not upload datasets, credentials, private paths, or model artifacts without confirming their redistribution terms.

## Development setup

```bash
conda create -n geocr-dev python=3.10 -y
conda activate geocr-dev

# Install a PyTorch build appropriate for your platform first.
pip install torch torchvision
pip install -e ".[dev]"
```

## Making a change

1. Create a focused branch from the current default branch.
2. Keep changes small and avoid unrelated formatting rewrites.
3. Add or update tests for behavior changes.
4. Run the checks below before opening a pull request.

```bash
ruff check train.py ultralytics/geocr tests/test_train.py
python -m compileall -q train.py ultralytics/geocr
python -m unittest discover -s tests -p test_train.py
```

Pull requests should explain the motivation, summarize the implementation, list the checks run, and document any
effect on training data, checkpoints, or reported metrics.

## Style

- Follow the existing Python style and keep lines at or below 120 characters.
- Prefer clear type annotations and concise docstrings for public behavior.
- Keep machine-specific paths out of committed YAML files; use repository-relative examples instead.
- Do not commit checkpoints, generated embeddings, experiment outputs, caches, or IDE settings.

By contributing, you agree that your contribution is licensed under the repository's AGPL-3.0 license.
