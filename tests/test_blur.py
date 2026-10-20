import numpy as np
import pytest
from portraitfx.core import background_blur

@pytest.mark.parametrize('scale',[1.,.5,.25])
def test_constant_background_preserved(scale):
    frame=np.full((31,47,3),120,np.uint8)
    alpha=np.zeros((31,47),np.float32);alpha[5:20,8:25]=1
    out=background_blur(frame,alpha,scale=scale)
    assert out.shape==frame.shape
    assert out.dtype==np.float32
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out,120/255,atol=1e-5)

def test_blur_scale_rejected():
    with pytest.raises(ValueError):background_blur(np.zeros((8,8,3),np.uint8),np.zeros((8,8)),scale=0)
