"""Validated NumPy-to-C++ boundary; no hidden fallback to Python."""
import ctypes
from functools import lru_cache
from pathlib import Path
import platform
import numpy as np

@lru_cache(maxsize=1)
def _library():
    path = Path(__file__).with_name('portrait_native.dll' if platform.system() == 'Windows' else 'portrait_native.so')
    if not path.exists():
        raise RuntimeError('C++ backend is not built. Run python build_native.py first.')
    lib = ctypes.CDLL(str(path))
    ptr = ctypes.POINTER(ctypes.c_float)
    lib.portrait_composite_v2.argtypes = [ptr, ptr, ptr, ptr, ctypes.c_size_t, ctypes.c_float]
    lib.portrait_composite_v2.restype = None
    return lib

def composite_cpp(foreground, alpha, background, strength=1.0):
    if not np.isfinite(strength) or not 0 <= strength <= 1: raise ValueError("strength must be in [0,1]")
    fg, a, bg = (np.asarray(x) for x in (foreground, alpha, background))
    if fg.ndim != 3 or fg.shape[2] != 3 or fg.shape != bg.shape or a.shape != fg.shape[:2]:
        raise ValueError('expected matching HxWx3 images and HxW alpha')
    fg, a, bg = (np.ascontiguousarray(x, dtype=np.float32) for x in (fg, a, bg))
    out = np.empty_like(fg)
    ptr = ctypes.POINTER(ctypes.c_float)
    _library().portrait_composite_v2(*(x.ctypes.data_as(ptr) for x in (fg, a, bg, out)), a.size, strength)
    return out
