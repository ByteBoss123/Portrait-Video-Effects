import numpy as np
import pytest
from portraitfx.core import composite,alpha_metrics,background_blur

def test_compositing_endpoints():
    fg=np.ones((8,8,3)); bg=np.zeros_like(fg)
    assert np.allclose(composite(fg,np.ones((8,8)),bg),fg)
    assert np.allclose(composite(fg,np.zeros((8,8)),bg),bg)
    assert np.allclose(composite(fg,np.full((8,8),.25),bg),.25)

def test_metric_known_error():
    gt=np.zeros((8,8),np.float32); pred=np.ones_like(gt)*.25
    m=alpha_metrics(pred,gt)
    assert m['mae']==.25 and m['mse']==.0625 and m['boundary_band_mae'] is None

def test_invalid_shapes():
    with pytest.raises(ValueError): alpha_metrics(np.zeros((4,4)),np.zeros((5,5)))

def test_nonfinite():
    with pytest.raises(ValueError): alpha_metrics(np.full((4,4),np.nan),np.zeros((4,4)))

def test_blur_no_foreground_bleed():
    frame=np.zeros((32,32,3),np.uint8); frame[8:24,8:24]=255
    a=np.zeros((32,32),np.float32); a[8:24,8:24]=1
    assert np.max(background_blur(frame,a))<.001
