import numpy as np
import torch
import sys
import os

# Add current directory to path to find the .so file
sys.path.insert(0, os.path.dirname(__file__))

# FIXED: Import the compiled module directly instead of relative import
try:
    # The .so file creates a module named 'core' - import it directly
    import core
    maximum_path_c = core.maximum_path_c
except ImportError as e:
    print(f"Error importing core module: {e}")
    print("Make sure core.cpython-38-x86_64-linux-gnu.so exists in the current directory")
    raise

def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
  path = np.zeros(neg_cent.shape, dtype=np.int32)

  t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
  t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)
