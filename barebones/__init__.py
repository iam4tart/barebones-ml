import os
import sys
import importlib
import importlib.util
import importlib.machinery
from pathlib import Path

try:
    import torch as _torch_check
    _torch_lib = Path(_torch_check.__file__).parent / "lib"
    if _torch_lib.exists() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(_torch_lib))
except Exception:
    pass

import torch

_here     = Path(__file__).parent
_libs_dir = _here / "libs"

def _load_extension(mod_name: str):
    short_name = mod_name.split(".")[-1]  

    pyd_files = list(_libs_dir.glob(f"{short_name}*.pyd")) + \
                list(_libs_dir.glob(f"{short_name}*.so"))

    if not pyd_files:
        raise ImportError(
            f"cannot find compiled extension '{short_name}' in {_libs_dir}\n"
            f"run: python setup.py build_ext --inplace"
        )

    _pyd = pyd_files[0]

    _loader = importlib.machinery.ExtensionFileLoader(mod_name, str(_pyd))
    _spec   = importlib.util.spec_from_file_location(
        mod_name, str(_pyd),
        loader=_loader,
        submodule_search_locations=[]
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[mod_name] = _mod
    _spec.loader.exec_module(_mod)
    return _mod

_ext_names = ["octree", "chamfer", "fps"]

for _name in _ext_names:
    try:
        _mod = _load_extension(f"barebones.libs.{_name}")
        globals()[_name] = _mod
    except ImportError as e:
        import warnings
        warnings.warn(f"barebones: could not load '{_name}': {e}")

__all__ = _ext_names