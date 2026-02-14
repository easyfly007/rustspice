from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def iter_netlists(root: Path, limit: int) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".cir", ".sp", ".spi", ".net", ".ckt"}:
            files.append(path)
        if len(files) >= limit:
            break
    return files


def build_sim_cli(repo_root: Path) -> Path:
    subprocess.run(
        ["cargo", "build", "-p", "sim-cli"],
        cwd=repo_root,
        check=True,
    )
    exe = repo_root / "target" / "debug" / "sim-cli"
    if sys.platform.startswith("win"):
        exe = exe.with_suffix(".exe")
    return exe


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_root = repo_root / "spice-datasets"

    # Check if spice-datasets exists in repo, if not try parent directory
    if not dataset_root.exists():
        # Try parent directory (../spice-datasets)
        parent_dataset = repo_root.parent / "spice-datasets"
        if parent_dataset.exists():
            dataset_root = parent_dataset
        else:
            print(f"spice-datasets not found: {dataset_root} or {parent_dataset}")
            return 2

    netlists = iter_netlists(dataset_root, limit=50)
    if not netlists:
        print("no netlists found in spice-datasets")
        return 2

    exe = build_sim_cli(repo_root)
    total = len(netlists)
    passed = 0
    failed = 0

    for netlist in netlists:
        result = subprocess.run(
            [str(exe), str(netlist)],
            cwd=repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            passed += 1
        else:
            failed += 1
            print(f"[FAIL] {netlist}")
            if result.stderr:
                print(result.stderr.strip())

    passrate = (passed / total) * 100.0
    print(f"total={total} passed={passed} failed={failed} passrate={passrate:.2f}%")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
