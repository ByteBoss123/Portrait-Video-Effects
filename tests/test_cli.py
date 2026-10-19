"""Regression: NumPy processing must not require the optional native backend."""
import json
from types import SimpleNamespace
import cv2
import numpy as np
from portraitfx import __main__ as cli
from portraitfx import native


def test_numpy_process_without_native_backend(tmp_path, monkeypatch):
    source=tmp_path/'input.mp4'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),10,(32,32))
    assert writer.isOpened()
    writer.write(np.full((32,32,3),127,np.uint8)); writer.release()
    class FakeRVM:
        def __init__(self,*args): pass
        def reset(self): pass
        def predict(self,frame,*args):
            return frame.astype(np.float32)/255,np.ones(frame.shape[:2],np.float32)
    def unavailable(): raise AssertionError('NumPy must not load native library')
    monkeypatch.setattr(cli,'RVM',FakeRVM)
    monkeypatch.setattr(native,'_library',unavailable)
    model=tmp_path/'model';model.write_bytes(b'test fixture')
    report=tmp_path/'report.json'
    cli.process(SimpleNamespace(input=str(source),output=str(tmp_path/'out.mp4'),
        width=32,warmup=0,strength=1.,backend='numpy',model=str(model),ratio=.5,
        threads=2,background=None,alpha_dir=None,cut_threshold=.35,
        independent=False,measure_memory=False,report=str(report)))
    result=json.loads(report.read_text())
    assert result['frames']==1
    assert result['compositing_backend']=='numpy'
