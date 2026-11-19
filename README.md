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

## Do before committing

```bash
ruff check .
ruff format .
mypy .
pytest
```

If mypy complains about third-party packages without type stubs, use `mypy --ignore-missing-imports .` or install the suggested `types-*` package.
