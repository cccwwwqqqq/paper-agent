# Architecture

The application is organized around a few clear layers:

- `settings.py` centralizes environment-derived configuration into a single `Settings` object.
- `bootstrap.py` assembles the runtime graph and supporting services.
- `services/` contains application orchestration such as chat handling, ingestion, and system lifecycle.
- `agents/` contains the LangGraph workflow, prompts, schemas, and retrieval policy.
- `storage/` contains Qdrant access, parent chunk storage, and workspace memory persistence.
- `models/` contains LLM and embedding factories.
- `parsers/` converts PDF or Markdown input into parsed documents.
- `ui/` contains the Gradio interface.

Runtime data is stored under `data/` by default, with compatibility fallbacks for legacy root-level directories when they already exist.

