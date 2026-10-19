import argparse, hashlib, json, platform, time
from pathlib import Path
import cv2
import numpy as np
import psutil
from .core import RVM, composite, background_blur, alpha_metrics

def process(a):
    if Path(a.input).resolve()==Path(a.output).resolve():
        raise ValueError('input and output must differ')
    if a.width < 32 or a.warmup < 0: raise ValueError('width >=32 and warmup >=0 required')
    blend = composite
    if a.backend == 'cpp':
        from .native import composite_cpp, _library
        _library()  # fail before processing if the native build is missing
        blend = composite_cpp
    cap=cv2.VideoCapture(a.input)
    ok,first=cap.read()
    if not ok: raise ValueError('cannot read input video')
    fps=cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=0: raise ValueError('invalid source FPS')
    width=a.width; height=max(2,round(first.shape[0]*width/first.shape[1]/2)*2)
    width=width//2*2
    first=cv2.resize(first,(width,height))
    model=RVM(a.model,a.ratio,a.threads)
    for _ in range(a.warmup): model.predict(first)
    model.reset()
    bg=None
    if a.background:
        bg=cv2.imread(a.background)
        if bg is None: raise ValueError('cannot read background')
        bg=cv2.resize(bg,(width,height)).astype(np.float32)/255
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(a.output,cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened(): raise RuntimeError('video writer failed')
    if a.alpha_dir: Path(a.alpha_dir).mkdir(parents=True,exist_ok=True)
    timings=[]; memory=[]; count=0; frame=first; previous=None; cuts=0
    try:
        while True:
            t0=time.perf_counter()
            if count:
                ok,frame=cap.read()
                if not ok: break
            t1=time.perf_counter(); frame=cv2.resize(frame,(width,height))
            thumb=cv2.resize(frame,(32,32)).astype(float)/255
            # Heuristic cut reset, deliberately documented rather than claimed perfect.
            if previous is not None and np.mean(abs(thumb-previous))>a.cut_threshold:
                model.reset(); cuts+=1
            previous=thumb
            t2=time.perf_counter(); fgr,alpha=model.predict(frame,a.independent)
            t3=time.perf_counter()
            background=bg if bg is not None else background_blur(frame,alpha)
            output=(blend(fgr,alpha,background)*255).round().astype('uint8')
            t4=time.perf_counter(); writer.write(output)
            if a.alpha_dir:
                cv2.imwrite(str(Path(a.alpha_dir)/f'{count:06d}.png'),(alpha*65535).round().astype('uint16'))
            t5=time.perf_counter()
            timings.append([(t1-t0)*1000,(t2-t1)*1000,(t3-t2)*1000,(t4-t3)*1000,(t5-t4)*1000,(t5-t0)*1000])
            try:
                if a.measure_memory:
                    memory.append(psutil.Process().memory_info().rss)
            except (psutil.Error, OSError):
                pass
            count+=1
    finally:
        cap.release(); writer.release()
    stats=np.array(timings)
    report={'compositing_backend':a.backend,'frames':count,'resolution':[width,height],'source_fps':fps,'warmup_frames':a.warmup,
        'ratio':a.ratio,'independent_frames':a.independent,'scene_cut_resets':cuts,
        'platform':platform.platform(),'processor':platform.processor(),'cpu_threads':a.threads,
        'model_bytes':Path(a.model).stat().st_size,'model_sha256':hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
        'sampled_peak_process_rss_bytes':max(memory) if memory else None,'timing_ms':{},
        'notes':['CPU only. RSS sampled after each frame, not true peak allocation.',
        'First frame decode excluded. Model load and warmup excluded. Encoder final flush excluded.',
        'Video output is silent. Blur is mask-based, not depth-based. No accuracy claim without ground truth.']}
    for i,key in enumerate(['decode','preprocess','inference','composite','write','pipeline']):
        report['timing_ms'][key]={'p50':float(np.percentile(stats[:,i],50)),'p95':float(np.percentile(stats[:,i],95))}
    Path(a.report).parent.mkdir(parents=True,exist_ok=True)
    Path(a.report).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

def evaluate(a):
    rows=[]
    pred=Path(a.pred); truth=Path(a.truth)
    names=sorted(p.name for p in truth.glob('*.png'))
    if not names: raise ValueError('no ground truth PNG mattes')
    if set(names)!=set(p.name for p in pred.glob('*.png')): raise ValueError('prediction/truth filenames differ')
    for name in names:
        arrays=[]
        for folder in [pred,truth]:
            x=cv2.imread(str(folder/name),cv2.IMREAD_UNCHANGED)
            if x is None or x.ndim!=2 or x.dtype not in (np.uint8,np.uint16): raise ValueError('8/16 bit grayscale PNG required')
            arrays.append(x.astype(np.float32)/np.iinfo(x.dtype).max)
        rows.append({'frame':name,**alpha_metrics(*arrays)})
    result={'frames':len(rows),'per_frame':rows,'macro_average':{k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in ['mae','mse','boundary_band_mae']}}
    Path(a.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result['macro_average']))

def main():
    parser=argparse.ArgumentParser(description='Portrait video effects: CPU RVM inference and alpha evaluation')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('process')
    p.add_argument('--backend', choices=['numpy','cpp'], default='numpy')
    for key in ['input','model','output']: p.add_argument('--'+key,required=True)
    p.add_argument('--measure-memory',action='store_true',help='Enable process RSS sampling on a host with a reliable process namespace'); p.add_argument('--background'); p.add_argument('--alpha-dir'); p.add_argument('--report',default='results/benchmark.json')
    p.add_argument('--width',type=int,default=640); p.add_argument('--ratio',type=float,default=.5)
    p.add_argument('--threads',type=int,default=2); p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--cut-threshold',type=float,default=.35); p.add_argument('--independent',action='store_true')
    p.set_defaults(fn=process)
    e=sub.add_parser('evaluate')
    for key in ['pred','truth','output']: e.add_argument('--'+key,required=True)
    e.set_defaults(fn=evaluate)
    a=parser.parse_args(); a.fn(a)
if __name__=='__main__': main()
