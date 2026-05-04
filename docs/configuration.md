# Configuration

The application loads configuration through `agentic_rag.settings.Settings`.

## Config Sources

1. Explicit environment variables
2. Root `.env`
3. Legacy fallback `project/.env` if root `.env` is absent
4. Built-in defaults

## Important Variables

- `DATA_DIR`: root directory for runtime data
- `LLM_PROVIDER`, `LLM_MODEL`, `LLM_TEMPERATURE`
- `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`
- `PDF_PARSER_BACKEND`: `pymupdf4llm` or `docling`
- `DOCLING_DO_FORMULA_ENRICHMENT`: set to `true` when ingesting formula-heavy PDFs and using the `docling` backend. This asks Docling to run formula recognition and convert mathematical expressions to LaTeX, which is slower but avoids many `formula-not-decoded` gaps.
- `DEFAULT_WORKSPACE_ID`
- `LANGFUSE_ENABLED`

The canonical template is `.env.example`.
