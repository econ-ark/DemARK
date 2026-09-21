"""Copy each notebook's LaTeX macros from its markdown cells into its MyST page frontmatter.

Three notebooks define their notation inline, as `$\\newcommand{\\Ex}{\\mathbb{E}}$` in a markdown
cell. Jupyter's MathJax honours that; MyST does not, because it renders each math node on its own
with the macros it was given in frontmatter. The site therefore showed 262 "Undefined control
sequence" badges, and every display on the LC-Model page came out as an error block.

MyST reads page frontmatter for a notebook from its top-level `metadata`, filtered to the page
frontmatter keys, one of which is `math`. This writes the macros there.

The cells stay the authored home, since they are what a reader running the notebook in Jupyter
gets. `metadata.math` is derived from them, so run this after changing a definition, and run
`--check` to fail when the two have parted:

    python3 scripts/sync-math-macros.py            # write
    python3 scripts/sync-math-macros.py --check    # report drift, exit 1

A later definition of the same name replaces an earlier one, as in LaTeX: LC-Model defines `\\pLvl`
twice, and the page reads in the second.
"""

import glob
import json
import logging
import re
import sys

logging.basicConfig(format="%(message)s", level=logging.INFO)

# \providecommand, since a cell that repeats a definition metadata.math already carries is what
# Temml refuses: it errors on \newcommand for a name it has, and passes over \providecommand.
# MathJax honours either, so the notebook still reads in Jupyter.
NEWCOMMAND = re.compile(r"\\(?:new|renew|provide)command\{\\(\w+)\}(?:\[\d\])?\s*\{")


def _balanced(text: str, brace: int) -> tuple[str, int]:
    """Return the body of the group opening at `brace`, and the index after its close."""
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : i], i + 1
    raise ValueError(f"unclosed brace at {brace}")


def macros_from_cells(notebook: dict) -> dict[str, str]:
    found: dict[str, str] = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        source = "".join(cell["source"])
        pos = 0
        while (match := NEWCOMMAND.search(source, pos)) is not None:
            body, pos = _balanced(source, match.end() - 1)
            found["\\" + match.group(1)] = body
    return found


def sync(path: str, check: bool) -> bool:
    """Return True when the file is already in sync (or has no macros to sync)."""
    with open(path, encoding="utf-8") as handle:
        notebook = json.load(handle)
    wanted = macros_from_cells(notebook)
    current = notebook["metadata"].get("math", {})
    if current == wanted:
        return True
    if check:
        logging.info(
            "%s: metadata.math has %d macro(s), the cells define %d",
            path,
            len(current),
            len(wanted),
        )
        for name in sorted(set(current) | set(wanted)):
            if current.get(name) != wanted.get(name):
                logging.info(
                    "    %-16s metadata %r  cells %r",
                    name,
                    current.get(name),
                    wanted.get(name),
                )
        return False
    if wanted:
        notebook["metadata"]["math"] = wanted
    else:
        notebook["metadata"].pop("math", None)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    logging.info("%s: wrote %d macro(s)", path, len(wanted))
    return True


def main(argv: list[str]) -> int:
    check = "--check" in argv
    paths = [a for a in argv if not a.startswith("--")] or sorted(
        glob.glob("notebooks/*.ipynb")
    )
    drifted = [p for p in paths if not sync(p, check)]
    if drifted:
        logging.info(
            "%d notebook(s) out of sync; run without --check to write", len(drifted)
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
