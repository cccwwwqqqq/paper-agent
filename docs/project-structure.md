# Project Structure

This repository uses a source-first Python package layout. Keep new application code in `src/agentic_rag` and treat the root-level runtime folders as local data, not source.

## Top-level Layout

- `src/agentic_rag/`: canonical application package.
- `tests/`: unit and integration tests.
- `docs/`: architecture, configuration, development, and migration notes.
- `eval/`: evaluation scripts, golden datasets, report templates, and ignored result outputs.
- `assets/`: checked-in product images used by docs or UI.
- `notebooks/`: exploratory notebooks.
- `data/`: default local runtime data root; ignored by git.
- `project/`: legacy compatibility package. Do not add new business logic here.
- `legacy/`: archived pre-migration files for reference.
- `agentic_rag/`: import-path shim that points local imports at `src/agentic_rag` when the package is not installed.

## Source Package Boundaries

Inside `src/agentic_rag`, keep modules grouped by responsibility:

- `agents/`: LangGraph graph, nodes, routing, prompts, tool wrappers, and agent schemas.
- `services/`: application orchestration such as chat, ingestion, retrieval, and runtime lifecycle.
- `storage/`: Qdrant, parent store, and workspace memory persistence.
- `models/`: LLM, embedding, and reranker factories.
- `parsers/`: document parser adapters and parser factory code.
- `ui/`: Gradio UI and UI-specific styling.
- `utils/`: small shared helpers with no business orchestration.

When a change spans layers, prefer dependencies flowing inward from entrypoints and services into focused helpers. Avoid imports from `project.*` in new code.

## Runtime Data

Runtime outputs should live under `data/` by default:

- `data/markdown_docs/`
- `data/parent_store/`
- `data/qdrant_db/`
- `data/workspace_memory/`
- `data/docling_artifacts/`

The settings layer still detects legacy root-level runtime directories when they already exist, but new local data should use `data/`.

## Generated Files

Keep these out of the repository:

- Python caches: `__pycache__/`, `.pytest_cache/`, `*.pyc`
- Package/build output: `*.egg-info/`, `build/`, `dist/`
- Local logs: `*.log`
- Runtime stores: `data/`, `qdrant_db/`, `markdown_docs/`, `parent_store/`, `workspace_memory/`
- Evaluation outputs: `eval/results/*.json`, `eval/results/*.md`

If a new tool creates repeatable local output, add it to `.gitignore` unless the output is meant to be reviewed and versioned.
