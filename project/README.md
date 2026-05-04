# Legacy Compatibility Layer

The canonical application now lives under `src/agentic_rag/`.

This `project/` directory is kept temporarily so that older commands such as:

```bash
python project/app.py
```

continue to work during the migration window.

For new development, use:

- `python -m agentic_rag.app`
- `python -m agentic_rag.cli`
- imports from `agentic_rag.*`

