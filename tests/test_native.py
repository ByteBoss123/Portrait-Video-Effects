import numpy as np
import pytest
from portraitfx.core import composite
from portraitfx.native import composite_cpp

@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.float16])
@pytest.mark.parametrize('strided', [False, True])
def test_native_parity(dtype, strided):
    rng = np.random.default_rng(42)
    fg = rng.uniform(-.2, 1.2, (24, 31, 3)).astype(dtype)
    bg = rng.random(fg.shape).astype(dtype)
    a = rng.uniform(-.5, 1.5, fg.shape[:2]).astype(dtype)
    if strided: fg, a, bg = fg[:, ::-1, ::-1], a[:, ::-1], bg[:, ::-1, ::-1]
    expected = composite(*(x.astype(np.float32) for x in (fg, a, bg)))
    np.testing.assert_allclose(composite_cpp(fg, a, bg), expected, atol=2e-7, rtol=0)

@pytest.mark.parametrize('a', [0., 1., .25])
def test_endpoints(a):
    fg = np.ones((3, 5, 3), np.float32)
    np.testing.assert_allclose(composite_cpp(fg, np.full((3,5), a), fg*0), a)

@pytest.mark.parametrize('shape', [(3, 5), (3, 5, 4)])
def test_reject_bad_buffers(shape):
    with pytest.raises(ValueError): composite_cpp(np.zeros(shape), np.zeros((3,5)), np.zeros(shape))

def test_nan_preserved():
    fg = np.zeros((1,1,3), np.float32)
    assert np.isnan(composite_cpp(fg, np.full((1,1), np.nan), fg)).all()
