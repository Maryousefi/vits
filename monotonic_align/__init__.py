# monotonic_align/__init__.py
import numpy as np
import torch
import sys
import os
import importlib.util

def maximum_path_python(neg_cent, mask):
    """Pure Python fallback for monotonic alignment (compatible with Python 3.8)."""
    device = neg_cent.device
    dtype = neg_cent.dtype
    
    # Convert to numpy
    neg_cent_np = neg_cent.data.cpu().numpy().astype(np.float32)
    mask_np = mask.data.cpu().numpy().astype(np.bool_)
    
    batch_size, t_t, t_s = neg_cent_np.shape
    path = np.zeros((batch_size, t_t, t_s), dtype=np.int32)
    
    for b in range(batch_size):
        # Get lengths from mask
        t_y = int(mask_np[b, :, 0].sum())
        t_x = int(mask_np[b, 0, :].sum())
        value_matrix = np.full((t_y, t_x), -1e9, dtype=np.float32)

        # Fill the value matrix
        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                if x == y:
                    value_matrix[y, x] = -1e9
                else:
                    value_matrix[y, x] = neg_cent_np[b, y-1, x] if y > 0 else -1e9

        # Forward DP
        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                v_prev = value_matrix[y-1, x-1] if (x > 0 and y > 0) else (-1e9 if x > 0 else 0.0)
                v_cur = value_matrix[y-1, x] if (y > 0) else -1e9
                value_matrix[y, x] += max(v_prev, v_cur)

        # Backtrace
        index = t_x - 1
        for y in range(t_y - 1, -1, -1):
            path[b, y, index] = 1
            if index != 0 and (index == y or (y > 0 and value_matrix[y-1, index] < value_matrix[y-1, index-1])):
                index -= 1

    return torch.from_numpy(path).to(device=device, dtype=dtype)

# ------------------------------
# Try importing the Cython version
# ------------------------------
current_dir = os.path.dirname(__file__)
so_file = os.path.join(current_dir, "core.cpython-38-x86_64-linux-gnu.so")

try:
    if os.path.exists(so_file):
        spec = importlib.util.spec_from_file_location("core", so_file)
        core_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_module)
        maximum_path_c = core_module.maximum_path_c

        def maximum_path_cython(neg_cent, mask):
            """Cython-accelerated version."""
            device = neg_cent.device
            dtype = neg_cent.dtype
            neg_cent_np = neg_cent.data.cpu().numpy().astype(np.float32)
            path_np = np.zeros(neg_cent_np.shape, dtype=np.int32)
            t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
            t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
            maximum_path_c(path_np, neg_cent_np, t_t_max, t_s_max)
            return torch.from_numpy(path_np).to(device=device, dtype=dtype)

        maximum_path = maximum_path_cython
        print("✓ Using Cython optimized maximum_path")
    else:
        raise ImportError(".so file not found")

except Exception as e:
    maximum_path = maximum_path_python
    print(f"✓ Using Python fallback for maximum_path: {e}")
