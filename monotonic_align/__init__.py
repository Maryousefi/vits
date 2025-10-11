import os
import sys
import numpy as np
import torch

# Add the current directory to sys.path so Python can find the compiled .so file
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import core
    maximum_path_c = core.maximum_path_c
except ImportError as e:
    print(f"Error importing core module: {e}")
    print(f"Make sure the .so file (core.cpython-38-x86_64-linux-gnu.so) exists in {current_dir}")
    raise

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
