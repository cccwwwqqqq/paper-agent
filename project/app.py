"""Legacy compatibility entrypoint.

Prefer: `python -m agentic_rag.app`
"""

def main():
    from agentic_rag.app import main as app_main

    app_main()


if __name__ == "__main__":
    main()
