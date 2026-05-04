# Migration Notes

## Entry points

- Old: `python project/app.py`
- New: `python -m agentic_rag.app`

## Configuration

- Old preferred config: `project/.env`
- New preferred config: `.env`

## Code layout

- Old implementation modules: `project/`
- New implementation modules: `src/agentic_rag/`

## Runtime data

The new default runtime root is `data/`. Existing root-level directories such as `markdown_docs/` and `qdrant_db/` are still detected automatically when present so existing local data is not broken during the migration window.

