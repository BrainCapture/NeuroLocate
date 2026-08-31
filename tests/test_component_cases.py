# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""Gate A: run every checked-in Tesseract regression case.

The cases in ``components/tesseracts/*/test_cases/*.json`` are executed here via
:meth:`Tesseract.test`, so they run without a container runtime. ``make test`` in
the cookiecutter layout runs the same files against *built images* with
``tesseract run <image> test @<file>`` — same files, both transports.

Regenerate the cases with ``python scripts/gen_test_cases.py``.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import pytest
from neurolayout.clients import open_component

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = REPO_ROOT / "components" / "tesseracts"

CASES = sorted(COMPONENTS.glob("*/test_cases/*.json"))


def test_every_component_has_regression_cases() -> None:
    """A component with no frozen case is silently unprotected."""
    components = {path.parent.parent.name for path in CASES}
    assert components == {"headfield", "proposal"}


@pytest.fixture(scope="module")
def case_clients():
    """One open client per component that owns a frozen case.

    Components are opened by name, with one `ExitStack` for the whole module,
    because opening `proposal` reads a 4.8 MB checkpoint and doing that once per
    case is pure latency.
    """
    with ExitStack() as stack:
        clients = {}

        def client(component: str):
            if component not in clients:
                clients[component] = stack.enter_context(open_component(component))
            return clients[component]

        yield client


@pytest.mark.parametrize("case_path", CASES, ids=lambda p: f"{p.parent.parent.name}/{p.stem}")
def test_regression_case(case_path: Path, case_clients) -> None:
    """Every frozen input/output pair must still hold exactly."""
    component = case_path.parent.parent.name
    case_clients(component).test(json.loads(case_path.read_text()))


def test_no_two_components_share_a_top_level_module_name() -> None:
    """Module names are effectively global once two components share a process.

    Each ``tesseract_api.py`` is loaded by path and puts *its own* directory on
    ``sys.path`` so it can import its siblings. When more than one component is
    opened in the same interpreter — which the test suite and
    ``neurolayout.clients.open_components`` both do — every component directory is
    on that one path, and a module name used by two components resolves to
    whichever loaded first.

    This is not hypothetical: two components in an earlier version of this
    project both shipped a ``model.py``, and opening them in one process raised
    ``cannot import name ... from 'model'`` while pointing at the other
    component's file. The image transport hid it, because in a container there is
    only ever one component.
    """
    owners: dict[str, list[str]] = {}
    for component in sorted(path for path in COMPONENTS.iterdir() if path.is_dir()):
        for module in component.glob("*.py"):
            if module.name == "tesseract_api.py":
                continue  # loaded by path, never by name
            owners.setdefault(module.stem, []).append(component.name)
    collisions = {name: sorted(who) for name, who in owners.items() if len(who) > 1}
    assert not collisions, (
        "these module names are shared between components and will shadow each "
        f"other when both are opened in one process: {collisions}"
    )


def test_declared_package_data_exists() -> None:
    """Every declared `package_data` source must exist.

    A file renamed in the tree but not in `tesseract_config.yaml` builds an image
    that is missing it, and the failure only appears at container runtime.
    """
    yaml = pytest.importorskip("yaml")
    for component in sorted(path for path in COMPONENTS.iterdir() if path.is_dir()):
        config = component / "tesseract_config.yaml"
        if not config.exists():
            continue
        entries = (
            yaml.safe_load(config.read_text()).get("build_config", {}).get("package_data")
            or []
        )
        for entry in entries:
            source = entry[0] if isinstance(entry, list) else entry
            assert (component / source).exists(), (
                f"{component.name}/tesseract_config.yaml declares package_data "
                f"'{source}', which does not exist"
            )
