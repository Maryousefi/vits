import os
import sys
import importlib.util
import numpy as np
import torch

# Add the current directory to sys.path so Python can find the compiled .so file
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Path to the compiled extension
so_file = os.path.join(current_dir, "core.cpython-38-x86_64-linux-gnu.so")
so_file_short = os.path.join(current_dir, "core.so")

# If needed, rename the .so file to a shorter name for cleaner import
if os.path.exists(so_file) and not os.path.exists(so_file_short):
    os.rename(so_file, so_file_short)

# Dynamically load the core module
spec = importlib.util.spec_from_file_location("core", so_file_short)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)

# Expose the C function
maximum_path_c = core.maximum_path_c

def maximum_path(neg_cent, mask):
    """Compute monotonic alignment path using the Cython core extension."""
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
    path = np.zeros(neg_cent.shape, dtype=np.int32)

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
    maximum_path_c(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)
