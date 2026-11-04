"""Reference results that generated hardware is compared against bit for bit."""
import numpy as np


def matvec_int(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Exact integer y = q @ x for signed 8-bit inputs: the golden output of a generated matvec module.

    The Verilog from `blaze.generate.generate_matrix` computes the same values with no rounding, so
    a simulator's y[i] must equal this result exactly (as a signed acc_width-bit integer).
    """
    q = np.asarray(q)
    x = np.asarray(x)
    if x.shape != (q.shape[1],):
        raise ValueError(f"x has shape {x.shape}, expected ({q.shape[1]},)")
    if x.size and (x.min() < -128 or x.max() > 127):
        raise ValueError("x must be signed 8-bit (-128..127)")
    return q.astype(np.int64) @ x.astype(np.int64)
