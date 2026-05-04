from __future__ import annotations

import shutil
from pathlib import Path


def clear_directory_contents(directory: Path) -> None:
    directory = Path(directory)
    if not directory.is_dir():
        return

    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

