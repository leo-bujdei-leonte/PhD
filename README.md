# PhD

## TODO
* uv setup instead of conda+pip
* 

## Setup

```bash
conda create -n phd python=3.12 -y
conda activate phd
pip install -r requirements.txt
```

## Linting and type checking

```bash
ruff check .
ruff format .
mypy .
```

Run all three before committing. If mypy complains about third-party packages without type stubs, use `mypy --ignore-missing-imports .` or install the suggested `types-*` package.
