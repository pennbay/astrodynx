# AstroDynX (adx)

A modern astrodynamics library powered by JAX.

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
