import os
import ctypes
import torch
from pathlib import Path

# 1. Locate the confirmed DLL directory
torch_lib_dir = Path(torch.__file__).parent / "lib"

if torch_lib_dir.exists():
    # Convert to absolute string for Windows
    dll_path = str(torch_lib_dir.resolve())
    
    # Add to search path
    os.add_dll_directory(dll_path)
    
    # 2. FORCE PRELOAD (The "Manual Handshake")
    # We load them in order of dependency: c10 first, then torch_cpu
    try:
        ctypes.CDLL(os.path.join(dll_path, "c10.dll"))
        ctypes.CDLL(os.path.join(dll_path, "torch_cpu.dll"))
    except Exception as e:
        print(f"Preload Warning: {e}")

# 3. LOAD YOUR LIBRARIES
libs_dir = Path(__file__).parent / "libs"
for pyd in libs_dir.glob("*.pyd"):
    try:
        torch.ops.load_library(str(pyd.resolve()))
    except Exception as e:
        print(f"Final failure loading {pyd.name}: {e}")

# 4. EXPORTS
from .fps import fps, mean_coverage, max_coverage
from .chamfer_distance import chamfer_distance
from .octree import (
    insert_points, query_range, nearest_neighbor, k_nearest_neighbor,
    remove_point, subdivide, redistribute, redistribute_query, save, load
)