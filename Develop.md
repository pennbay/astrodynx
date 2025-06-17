# Develop Guide

Initialize
```bash
pip install -e .[dev]
pre-commit install
```
After code change
```bash
pytest
```
Push code to source repo
```bash
git add .
pre-commit
cz c
git push -u origin main --tags
```


## Git commonds for forking
Add upstream
```bash
git remote add upstream https://github.com/pennbay/astrodynx.git
```

Pull upstream
```bash
git checkout main
git fetch upstream
```

Rebase to upstream
```bash
git stash
git rebase upstream/main
git stash pop
```

Force push
```bash
git push -f origin main
```
