"""Paired effect timings on cached RVM outputs; quality relative to full-scale blur."""
import argparse,json,time,platform,hashlib
from pathlib import Path
import cv2,numpy as np
from portraitfx.core import RVM,background_blur,composite
from portraitfx.native import composite_cpp

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--model',required=True)
    p.add_argument('--output',default='results/effects_comparison.json')
    p.add_argument('--width',type=int,default=480);p.add_argument('--frames',type=int,default=30)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--ratio',type=float,default=.5)
    a=p.parse_args()
    if min(a.width,a.frames,a.repeats)<1:p.error('positive width, frames and repeats required')
    cv2.setNumThreads(2);rvm=RVM(a.model,a.ratio)
    cap=cv2.VideoCapture(a.input);cached=[]
    try:
        while len(cached)<a.frames:
            ok,im=cap.read()
            if not ok:break
            im=cv2.resize(im,(a.width,max(2,round(im.shape[0]*a.width/im.shape[1]/2)*2)))
            fg,alpha=rvm.predict(im);cached.append((im,fg,alpha))
    finally:cap.release()
    if not cached:raise ValueError('no readable frames')
    configs=[(name,scale) for name in ('numpy','cpp') for scale in (1.,.5,.25)]
    samples={str(c):[] for c in configs};errors={str(c):[] for c in configs}
    for im,fg,alpha in cached:
        ref=composite(fg,alpha,background_blur(im,alpha))
        for name,scale in configs:
            fn=composite if name=='numpy' else composite_cpp
            out=fn(fg,alpha,background_blur(im,alpha,scale=scale))
            errors[str((name,scale))].append(float(np.mean(abs(out-ref))))
    # Warm up every path; alternate order to reduce ordering bias.
    im,fg,alpha=cached[0]
    for name,scale in configs:
        (composite if name=='numpy' else composite_cpp)(fg,alpha,background_blur(im,alpha,scale=scale))
    for repeat in range(a.repeats):
        for i,(im,fg,alpha) in enumerate(cached):
            order=configs if (repeat+i)%2==0 else configs[::-1]
            for name,scale in order:
                fn=composite if name=='numpy' else composite_cpp
                t=time.perf_counter_ns()
                bg=background_blur(im,alpha,scale=scale)
                tb=time.perf_counter_ns();fn(fg,alpha,bg);te=time.perf_counter_ns()
                samples[str((name,scale))].append([(tb-t)/1e6,(te-tb)/1e6,(te-t)/1e6])
    report=dict(platform=platform.platform(),frames=len(cached),repeats=a.repeats,
        input_sha256=hashlib.sha256(Path(a.input).read_bytes()).hexdigest(),
        config=vars(a),opencv_threads=cv2.getNumThreads(),results=[],
        scope='Cached real-frame effects only, excluding inference/IO. RGB MAE is deviation from full-scale NumPy blur, NOT ground-truth quality.')
    for name,scale in configs:
        key=str((name,scale));v=np.array(samples[key])
        report['results'].append(dict(backend=name,blur_scale=scale,rgb_mae_vs_reference=float(np.mean(errors[key])),
            timing_ms={label:dict(p50=float(np.median(v[:,i])),p95=float(np.percentile(v[:,i],95)))
                       for i,label in enumerate(('blur','blend','effects'))},samples_ms=samples[key]))
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report['results'],indent=2))
if __name__=='__main__':main()
