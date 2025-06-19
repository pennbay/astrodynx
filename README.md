[![PyPI - Version](https://img.shields.io/pypi/v/astrodynx)](https://pypi.org/project/astrodynx/)
[![GitHub License](https://img.shields.io/github/license/adxorg/astrodynx)](https://github.com/adxorg/astrodynx/blob/main/LICENSE)
[![Github CI](https://github.com/adxorg/astrodynx/actions/workflows/ci.yml/badge.svg)](https://github.com/adxorg/astrodynx/actions/workflows/ci.yml)
[![Deploy Docs](https://github.com/adxorg/astrodynx/actions/workflows/docs.yml/badge.svg)](https://github.com/adxorg/astrodynx/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/adxorg/astrodynx/graph/badge.svg?token=azxgWzPIIU)](https://codecov.io/gh/adxorg/astrodynx)


# AstroDynX (adx)

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

## Build Docs
```bash
pip install -e .[docs]
sphinx-build -b html docs docs/_build/html
```

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.
