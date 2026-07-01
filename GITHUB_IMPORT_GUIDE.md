# Import this zip into GitHub

## 1. Extract

```bash
unzip hermes-unified-agent-github-ready.zip
cd hermes-unified-agent
```

## 2. Initialize Git

```bash
git init
git add .
git commit -m "Initial Hermes unified agent source"
```

## 3. Create GitHub repo and push

```bash
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

## 4. Validate locally

```bash
python -m pytest tests/unified/test_unified_core.py -q
```

## Important

This repository vendors OmniAgent and AgentScope. Read `THIRD_PARTY_NOTICES.md`
before distributing publicly.
