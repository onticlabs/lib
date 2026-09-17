"""Uniform errors for missing optional dependencies.

Every optional import across ``ontic_lib`` / ``ontic_nn`` / ``ontic_data`` raises through
here, so a missing extra always reports the same three things: what needed it, which
distribution provides it, and the exact command that repairs the environment::

    TACO needs the 'torchcodec' video backend.
    Repair: uv add --dev "ontic-data[video]"

The ``--dev`` form is what this workspace needs: the packages are dev-installed members of
the root project. A downstream experiment that depends on them normally drops ``--dev``.

Some extras only cover a research package's *pip-installable* support deps, because the
package itself is not on PyPI. Pass ``repo_url`` for those and the repair grows a second
line pointing at the repository.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["missing_dependency", "optional_import", "repair_command"]

#: How the repair line is rendered. Workspace members are dev dependencies of the root.
_UV_ADD = 'uv add --dev "{package}[{extra}]"'


def repair_command(package: str, extra: str) -> str:
    """The ``uv add`` command that installs ``package``'s ``extra``."""
    return _UV_ADD.format(package=package, extra=extra)


def missing_dependency(
    what: str,
    *,
    package: str,
    extra: str,
    needs: str | None = None,
    repo_url: str | None = None,
    error_type: type[Exception] = ImportError,
) -> Exception:
    """Build the error for a missing optional dependency.

    ``what`` names the caller ("TACO", "splats.rendering"), ``needs`` the module that was
    absent, and ``package``/``extra`` the distribution and extra that provide it.
    ``repo_url`` marks a research package that the extra cannot install on its own.

    ``error_type`` stays ``ImportError`` for a failed import. Call sites whose *module*
    imports fine and that only fail when a capability is used (``splats.rendering``,
    ``splats.cute``) keep raising ``RuntimeError``.
    """
    head = f"{what} needs {needs}." if needs else f"{what} needs an optional dependency."
    lines = [head, f"Repair: {repair_command(package, extra)}"]
    if repo_url is not None:
        lines.append(f"        then install {repo_url} (clone it, or pip install --no-deps <clone>)")
    return error_type("\n".join(lines))


def optional_import(
    module: str,
    *,
    package: str,
    extra: str,
    what: str,
    repo_url: str | None = None,
) -> Any:
    """``importlib.import_module`` that reports a missing module as a repairable extra."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        needs = f"the `{module.split('.')[0]}` package"
        raise missing_dependency(
            what, package=package, extra=extra, needs=needs, repo_url=repo_url
        ) from e
