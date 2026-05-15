from __future__ import annotations

import shutil
from pathlib import Path


def init_demo(base_dir: str = ".") -> None:
    base = Path(base_dir)
    source_data = Path(__file__).resolve().parents[1] / "data"
    if not source_data.exists():
        raise FileNotFoundError(
            "Demo source data was not found. Run this command from the repository checkout, "
            "or use the checked-in data/ directory as your demo fixture."
        )

    target_data = base / "data"
    if source_data.resolve() == target_data.resolve():
        print(f"Demo files already available under {target_data.resolve()}")
        return

    for source_path in source_data.rglob("*"):
        relative_path = source_path.relative_to(source_data)
        target_path = target_data / relative_path
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)

    print(f"Demo files created under {target_data.resolve()}")
