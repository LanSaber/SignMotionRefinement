#!/usr/bin/env python
"""Run SignMotionRefinement commands directly from a source checkout."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

COMMANDS = {
    "complete": "sign_motion_refinement.cli.complete",
    "evaluate-jerk": "sign_motion_refinement.cli.evaluate_jerk",
    "evaluate-meta": "sign_motion_refinement.cli.evaluate_meta",
    "run-bounded-pilot": "sign_motion_refinement.cli.run_bounded_pilot",
    "train-mask-aware": "sign_motion_refinement.cli.train_mask_aware",
    "train-guava-only": "sign_motion_refinement.cli.train_guava_only",
    "visualize-completion": "sign_motion_refinement.visualization.completion_compare",
    "visualize-linear-siren": "sign_motion_refinement.visualization.linear_siren_jerk",
    "visualize-meta-jerk": "sign_motion_refinement.visualization.meta_jerk",
}


def usage() -> str:
    commands = "\n".join(f"  {name}" for name in COMMANDS)
    return (
        "usage: python smr.py <command> [arguments]\n\n"
        "commands:\n"
        f"{commands}\n\n"
        "Use `python smr.py <command> --help` for command-specific options."
    )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(usage())
        return
    command = sys.argv[1]
    if command not in COMMANDS:
        raise SystemExit(f"Unknown command {command!r}.\n\n{usage()}")
    sys.argv = [f"smr {command}", *sys.argv[2:]]
    importlib.import_module(COMMANDS[command]).main()


if __name__ == "__main__":
    main()
