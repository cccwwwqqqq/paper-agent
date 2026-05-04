from pathlib import Path

_source_package = Path(__file__).resolve().parents[1] / "src" / "agentic_rag"
if str(_source_package) not in __path__:
    __path__.append(str(_source_package))

