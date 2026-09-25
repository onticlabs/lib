"""Every ontic_lib / ontic_nn module must import with core dependencies only.

Optional packages are blocked in a subprocess (``sys.modules[name] = None``) after the core
deps are loaded; each module is then imported in that process and reported individually.
"""

import importlib.util
import json
import pkgutil
import subprocess
import sys
import textwrap

import pytest

BLOCKLIST = [
    "e3nn",
    "gsplat",
    "cutlass",
    "cuda",
    "safetensors",
    "plyfile",
    "wandb",
    "spconv",
    "torch_scatter",
    "flash_attn",
    "timm",
    "addict",
    "pointops",
    "point_rope_cuda",
    "serialize_cuda",
    "point_rope",
]

_SCRIPT = textwrap.dedent(
    """
    import importlib, json, sys, traceback
    import numpy, roma, torch  # core deps first, then block the optional ones
    blocklist, modules = json.loads(sys.argv[1]), json.loads(sys.argv[2])
    sys.modules.update({name: None for name in blocklist})
    results = {}
    for name in modules:
        try:
            importlib.import_module(name)
            results[name] = None
        except BaseException:
            results[name] = traceback.format_exc()
    print(json.dumps(results))
    """
)


def _modules(package_name):
    if importlib.util.find_spec(package_name) is None:
        return []
    package = importlib.import_module(package_name)
    names = [package_name]
    for info in pkgutil.walk_packages(package.__path__, package_name + "."):
        names.append(info.name)
    return names


MODULES = _modules("ontic_lib") + _modules("ontic_nn")


@pytest.fixture(scope="session")
def import_results():
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT, json.dumps(BLOCKLIST), json.dumps(MODULES)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_without_optional_deps(module, import_results):
    error = import_results[module]
    assert error is None, f"{module} failed to import with optional deps blocked:\n{error}"
