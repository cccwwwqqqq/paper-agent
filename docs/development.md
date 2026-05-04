# Development

## Install

```bash
pip install -e .[dev,eval]
```

## Run tests

```bash
python -m pytest -q
```

`unittest` discovery remains available for compatibility:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

If `.venv` was created with an interpreter that no longer exists, rebuild it:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .[dev,eval]
```

Evaluation runs enforce gradual quality gates by default. Use `--no-fail-on-threshold` when you only want a report:

```bash
python -m agentic_rag.cli eval --skip-judge --no-fail-on-threshold
```

## Layout

- Production code: `src/agentic_rag`
- Tests: `tests`
- Evaluations: `eval`
- Runtime data: `data`

For a fuller directory map and source package boundaries, see [Project Structure](project-structure.md).

## Notes

- `project/` is a compatibility layer and should not receive new business logic.
- New code should import from `agentic_rag`, not from legacy module paths.
- Generated files such as logs, caches, build output, and local vector stores should stay untracked.
