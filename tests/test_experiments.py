import numpy as np
import pytest
from portraitfx.core import composite
from portraitfx.native import composite_cpp
from experiments.reconstruct import reconstruct

@pytest.mark.parametrize('strength',[0,.25,1])
def test_strength(strength):
    fg=np.ones((3,4,3),np.float32); bg=np.zeros_like(fg); a=np.zeros((3,4),np.float32)
    for fn in [composite,composite_cpp]:
        np.testing.assert_allclose(fn(fg,a,bg,strength),1-strength)

@pytest.mark.parametrize('strength',[-1,2,float('nan')])
def test_bad_strength(strength):
    for fn in [composite,composite_cpp]:
        with pytest.raises(ValueError): fn(np.zeros((2,2,3)),np.zeros((2,2)),np.zeros((2,2,3)),strength)

def test_known_camera_geometry():
    rng=np.random.default_rng(7); X=rng.uniform([-1,-1,4],[1,1,8],(100,3))
    K=np.array([[700,0,320],[0,700,240],[0,0,1.]])
    def proj(X):
        q=(K@X.T).T; return q[:,:2]/q[:,2,None]
    _,r=reconstruct(proj(X),proj(X+[.7,0,0]),K)
    assert r['points']>80
    assert r['mean_reprojection_error_px']<.01
    np.testing.assert_allclose(r['R_world_to_camera2'],np.eye(3),atol=.001)
    np.testing.assert_allclose(r['t_world_to_camera2_unit_scale'],[1,0,0],atol=.001)

def test_insufficient_matches():
    with pytest.raises(ValueError): reconstruct(np.zeros((3,2)),np.zeros((3,2)),np.eye(3))

def test_manifest_rejects_cross_split_group(tmp_path):
    import cv2,json
    from experiments.data import load_manifest
    cv2.imwrite(str(tmp_path/'a.png'),np.zeros((4,4,3),np.uint8))
    cv2.imwrite(str(tmp_path/'mask.png'),np.zeros((4,4),np.uint8))
    rows=[dict(image='a.png',alpha='mask.png',group='same_person',license='fixture',split=s) for s in ['train','test']]
    p=tmp_path/'m.json';p.write_text(json.dumps({'samples':rows}))
    with pytest.raises(ValueError,match='group crosses splits'):load_manifest(p)

def test_manifest_rejects_duplicate_image(tmp_path):
    import cv2,json
    from experiments.data import load_manifest
    cv2.imwrite(str(tmp_path/'a.png'),np.zeros((4,4,3),np.uint8))
    cv2.imwrite(str(tmp_path/'mask.png'),np.zeros((4,4),np.uint8))
    rows=[dict(image='a.png',alpha='mask.png',group=s,license='fixture',split=s) for s in ['train','test']]
    p=tmp_path/'m.json';p.write_text(json.dumps({'samples':rows}))
    with pytest.raises(ValueError,match='duplicate image'):load_manifest(p)
