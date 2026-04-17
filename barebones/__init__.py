import os
import torch
from pathlib import Path

torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib_path):
    os.add_dll_directory(torch_lib_path)

libs_dir = Path(__file__).parent / "libs"

pyd_path = str(next(libs_dir.glob("fps*.pyd")))
torch.ops.load_library(pyd_path)