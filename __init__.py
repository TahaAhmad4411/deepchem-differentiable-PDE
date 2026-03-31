"""Extend the installed deepchem package with local subpackages."""

import importlib as _importlib
import os as _os
import sys as _sys

_local_dir = _os.path.dirname(_os.path.abspath(__file__))
_parent_dir = _os.path.dirname(_local_dir)

# Remove this local module so the installed deepchem can be found.
_self = _sys.modules.pop(__name__)
_saved_path = _sys.path[:]
_sys.path = [
    p for p in _sys.path
    if _os.path.realpath(p) != _os.path.realpath(_parent_dir)
]

_real = _importlib.import_module(__name__)

# Restore sys.path and re-register the real package.
_sys.path = _saved_path
_sys.modules[__name__] = _real

# Let the installed package also search the local directory.
if _local_dir not in _real.__path__:
    _real.__path__.insert(0, _local_dir)

# Eagerly patch installed sub-packages that have local extensions.
for _subpkg in ("data", "feat", "models", "models.torch_models"):
    _full = "deepchem." + _subpkg
    _local_sub = _os.path.join(_local_dir, *_subpkg.split("."))
    if not _os.path.isdir(_local_sub):
        continue
    _mod = _sys.modules.get(_full)
    if _mod is None:
        try:
            _mod = _importlib.import_module(_full)
        except ImportError:
            continue
    if hasattr(_mod, "__path__") and _local_sub not in _mod.__path__:
        _mod.__path__.insert(0, _local_sub)
