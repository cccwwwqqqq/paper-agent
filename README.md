# Agentic RAG for Dummies

A modular Agentic RAG application built with LangGraph, Gradio, and Qdrant.

This repository now follows an application-first layout:

- application code lives in `src/agentic_rag/`
- runtime data lives under `data/`
- long-form docs live in `docs/`
- evaluation tooling stays in `eval/`
- `project/` remains as a temporary compatibility shim

## Quick Start

1. Create a virtual environment and install the package:

```bash
pip install -e .[dev,eval]
```

2. Copy `.env.example` to `.env` and fill in your model settings.

3. Start the UI:

```bash
python -m agentic_rag.app
```

Or use the CLI:

```bash
python -m agentic_rag.cli serve
```

## Common Commands

Start the app:

```bash
python -m agentic_rag.app
```

Ingest documents into a workspace:

```bash
python -m agentic_rag.cli ingest --workspace demo --files path/to/paper.pdf
```

Run the evaluation suite:

```bash
python -m agentic_rag.cli eval
```

Run tests:

```bash
python -m pytest -q
```

If the local virtual environment points at a missing interpreter, rebuild it before testing:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .[dev,eval]
```

## Repository Guide

- [Architecture](docs/architecture.md)
- [Project Structure](docs/project-structure.md)
- [Configuration](docs/configuration.md)
- [Development](docs/development.md)
- [Migration Notes](docs/migration.md)

## Compatibility Notes

- Preferred config file: repository root `.env`
- Preferred entrypoint: `python -m agentic_rag.app`
- Legacy `python project/app.py` still works as a shim during the migration window
