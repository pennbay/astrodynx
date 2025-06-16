![GitHub License](https://img.shields.io/github/license/pennbay/astrodynx)


# AstroDynX

A modern astrodynamics library powered by JAX: differentiate, vectorize, JIT to GPU/TPU, and more.

## Features
- JAX-based fast computation
- Pre-commit code style and type checks
- Continuous testing
- Automated versioning and changelog
- GitHub Actions for CI/CD

## Installation
```bash
pip install astrodynx
```

## Usage
Check version
```python
import astrodynx as adx
print(adx.__version__)
```


## Development
Initialize
```bash
pip install -e .[dev]
pre-commit install
```
After code change
```bash
pytest
pre-commit
```
