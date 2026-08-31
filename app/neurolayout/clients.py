# Copyright 2026 NeuroLocate contributors
# SPDX-License-Identifier: Apache-2.0
"""Construction of the Tesseract clients the orchestrator composes.

Three transports are supported, all yielding the same
:class:`tesseract_core.Tesseract` object so nothing downstream can tell them
apart:

``local``
    Import ``tesseract_api.py`` directly via
    :meth:`Tesseract.from_tesseract_api`. No Docker. This is the transport used
    by the test suite so the gradient gates run anywhere, and it requires the
    component dependencies to be installed in the calling environment.

``image``
    Built container images (``tesseract build``), served on demand. This is the
    real deployment mode, where each component carries its own dependency
    tree: OpenMEEG's compiled stack in one image, PyTorch in the other.

``url``
    Already-serving Tesseracts, addressed by URL.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Literal

from tesseract_core import Tesseract

__all__ = [
    "Transport",
    "COMPONENT_ROOT",
    "IMAGE_NAMES",
    "HYBRID_COMPONENTS",
    "open_component",
    "open_components",
]

Transport = Literal["local", "image", "url"]

#: Repository-relative location of the Tesseract components.
COMPONENT_ROOT = Path(__file__).resolve().parents[2] / "components" / "tesseracts"

#: Image names produced by `make build` (from each tesseract_config.yaml).
IMAGE_NAMES = {
    "headfield": "neurolayout_headfield",
    "proposal": "neurolayout_proposal",
}

#: The two components on the scientific path, in gradient order: an epoch and a
#: parameter vector enter ``proposal``, its source set enters ``headfield``, and
#: the cotangent comes back through both.
HYBRID_COMPONENTS: tuple[str, ...] = ("proposal", "headfield")


def _api_path(component: str) -> Path:
    path = COMPONENT_ROOT / component / "tesseract_api.py"
    if not path.exists():
        raise FileNotFoundError(f"no tesseract_api.py for component {component!r}: {path}")
    return path


@contextmanager
def open_component(
    component: str,
    transport: Transport = "local",
    *,
    url: str | None = None,
) -> Iterator[Tesseract]:
    """Open a single Tesseract for the duration of the context.

    Source localization only needs ``headfield``; opening the PyTorch
    proposal alongside it would cost seconds and prove nothing.

    Args:
        component: ``"headfield"`` or ``"proposal"``.
        transport: One of ``"local"``, ``"image"``, ``"url"``.
        url: Required for ``transport="url"``.

    Yields:
        The opened :class:`tesseract_core.Tesseract`.
    """
    if transport == "local":
        with Tesseract.from_tesseract_api(_api_path(component)) as tesseract:
            yield tesseract
    elif transport == "image":
        with Tesseract.from_image(IMAGE_NAMES[component]) as tesseract:
            yield tesseract
    elif transport == "url":
        if not url:
            raise ValueError("transport='url' needs a url")
        yield Tesseract.from_url(url)
    else:
        raise ValueError(f"unknown transport {transport!r}")


@contextmanager
def open_components(
    names: Iterable[str],
    transport: Transport = "local",
    *,
    urls: dict[str, str] | None = None,
) -> Iterator[dict[str, Tesseract]]:
    """Open several Tesseracts at once, keyed by component name.

    Args:
        names: Component names, e.g. :data:`HYBRID_COMPONENTS`.
        transport: One of ``"local"``, ``"image"``, ``"url"``.
        urls: Required for ``transport="url"``; must cover every requested name.

    Yields:
        ``{name: Tesseract}``, all live for the duration of the context.
    """
    wanted = tuple(names)
    with ExitStack() as stack:
        if transport == "url":
            if not urls or not set(wanted) <= set(urls):
                missing = sorted(set(wanted) - set(urls or {}))
                raise ValueError(f"transport='url' needs urls for {missing}")
            yield {name: Tesseract.from_url(urls[name]) for name in wanted}
            return
        yield {
            name: stack.enter_context(open_component(name, transport)) for name in wanted
        }
