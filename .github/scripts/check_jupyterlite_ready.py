"""Fail while the released econ-ark cannot install under Pyodide.

In-page execution runs notebooks in the reader's browser, where micropip
resolves econ-ark from PyPI. Three of HARK's dependencies have no WebAssembly
build, so a release is usable in that setting only once it gates them on the
emscripten platform marker. This gate reads the published metadata for that
property. A pinned version number would go stale the next time HARK reorganises
its requirements.

Pass a path to a saved PyPI JSON payload as argv[1] to check a fixture. The
rejection test uses one fixture per outcome.
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.request

METADATA_URL = "https://pypi.org/pypi/econ-ark/json"

# None of these three publishes a WebAssembly wheel, so each has to be gated
# out of the dependency set when the platform is emscripten.
NATIVE_ONLY = ("numba", "interpolation", "quantecon")

logger = logging.getLogger(__name__)


def requirement_name(requirement: str) -> str:
    """Return the bare distribution name from a requires_dist entry."""
    name = requirement.split(";", 1)[0]
    for separator in ("[", "(", "<", ">", "=", "!", "~", " "):
        name = name.split(separator, 1)[0]
    return name.strip().replace("_", "-").lower()


def marker_of(requirement: str) -> str:
    """Return the environment marker of a requires_dist entry, or an empty string."""
    _, _, marker = requirement.partition(";")
    return marker.strip()


def load_metadata(source: str | None) -> dict:
    if source is None:
        with urllib.request.urlopen(METADATA_URL, timeout=30) as response:
            return json.load(response)
    with open(source, encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str]) -> int:
    logging.basicConfig(format="%(message)s", level=logging.INFO, stream=sys.stdout)

    metadata = load_metadata(argv[1] if len(argv) > 1 else None)
    version = metadata["info"]["version"]
    requirements = metadata["info"].get("requires_dist") or []

    ungated = []
    absent = []
    for target in NATIVE_ONLY:
        matches = [r for r in requirements if requirement_name(r) == target]
        if not matches:
            absent.append(target)
            continue
        if not any("emscripten" in marker_of(r) for r in matches):
            ungated.append(target)

    logger.info("econ-ark on PyPI: %s", version)
    for target in NATIVE_ONLY:
        state = "absent"
        if target in ungated:
            state = "present, no emscripten marker"
        elif target not in absent:
            state = "gated on emscripten"
        logger.info("  %-14s %s", target, state)

    if ungated:
        logger.error(
            "\nFAIL: econ-ark %s declares %s without an emscripten marker, so "
            "micropip cannot install it under Pyodide and the site's Run button "
            "fails at import HARK. Merging the JupyterLite switch now would "
            "publish a broken Run button.\n"
            "This gate clears itself once a release carrying econ-ark/HARK#1819 "
            "reaches PyPI.",
            version,
            ", ".join(ungated),
        )
        return 1

    logger.info("\nPASS: econ-ark %s gates every native-only dependency.", version)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
