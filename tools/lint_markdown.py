#!/usr/bin/env python3
"""Enhanced Markdown linter wrapper used by Makefile and pre-commit.

Usage::
    python tools/lint_markdown.py [path ...] [--fix] [--verbose] [--test-env] [--guidelines]

The script searches for a working ``markdownlint`` binary (Homebrew, npx, or global
npm install) and invokes it with sensible defaults.  Exit status is non-zero if
lint errors are found or if markdownlint is unavailable.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def find_markdownlint() -> str | None:
    """Return path to a working markdownlint-cli binary or ``None``."""
    candidates = [
        "/opt/homebrew/bin/markdownlint",  # Homebrew on Apple Silicon
        shutil.which("markdownlint"),       # global npm install
        shutil.which("markdownlint-cli"),
    ]
    for path in candidates:
        if path and Path(path).is_file():
            return path
    # As fallback, try npx which downloads into cache on first run
    if shutil.which("npx"):
        return "npx markdownlint-cli"
    return None


def run(cmd: list[str]) -> int:
    """Run *cmd* and return its exit code, streaming output."""
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    proc.communicate()
    return proc.returncode


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Markdown linter wrapper")
    parser.add_argument("paths", nargs="*", default=["."], help="files or directories to lint")
    parser.add_argument("--fix", action="store_true", help="auto-fix fixable issues")
    parser.add_argument("--verbose", action="store_true", help="show underlying markdownlint command")
    parser.add_argument("--test-env", action="store_true", help="print environment diagnostics and exit")
    parser.add_argument("--guidelines", action="store_true", help="print formatting guidelines")
    args = parser.parse_args()

    if args.guidelines:
        print("Refer to prompts/markdown-linting-daily-workflow.md for guidelines.")
        sys.exit(0)

    mdlint = find_markdownlint()
    if mdlint is None:
        print("ERROR: markdownlint-cli not found. See prompts/markdown-linting-daily-workflow.md for installation steps.", file=sys.stderr)
        sys.exit(2)

    if args.test_env:
        print(f"markdownlint binary: {mdlint}")
        sys.exit(0)

    cmd = []
    if mdlint.startswith("npx "):
        cmd = mdlint.split()
    else:
        cmd = [mdlint]

    if args.fix:
        cmd.append("--fix")

    cmd.extend(args.paths)

    if args.verbose:
        print("Running:", " ".join(cmd))

    exit_code = run(cmd)
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 