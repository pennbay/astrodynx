# Develop Guide

## Clone repository
Fork the repo and clone it to your local machine.
```bash
git clone <your-fork-url>
cd astrodynx
git remote add upstream https://github.com/pennbay/astrodynx.git
```

## Setup development environment
Install development dependencies.
```bash
pip install -e .[dev]
pre-commit install
```

Test the installation.
```bash
pytest
```

## Development workflow
Before making any changes, ensure your local repository is up to date with the upstream repository.
```bash
git fetch upstream
git checkout main
git rebase upstream/main
```
Make changes to the codebase, then test your changes.
```bash
pytest
```

## Commit changes

Stash any uncommitted changes
```bash
git stash
```

Before committing, ensure your local repository is still up to date with the upstream repository.
```bash
git fetch upstream
git checkout main
git rebase upstream/main
```
If there are conflicts, resolve them and continue the rebase.
```bash
git rebase --continue
```
Then, pop your stashed changes
```bash
git stash pop
```
If you have made changes to the code, run the tests again to ensure everything is working.
```bash
pytest
```
If all tests pass, you can proceed to commit your changes.
```bash
git add .
pre-commit
cz c
```
Push your changes to your fork
```bash
git push -u origin main --tags
```

## Create a pull request
Go to the original repository on GitHub and create a pull request from your fork's `main` branch to the original repository's `main` branch. Provide a clear description of your changes and any relevant information.

## Check your github actions
After creating the pull request, GitHub Actions will automatically run tests and checks on your code. Ensure that all checks pass before the pull request can be merged.
