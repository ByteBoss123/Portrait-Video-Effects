"""Paired, alternating blend benchmark. No inference, blur, or video I/O."""
import argparse, json, platform, time
from pathlib import Path
import numpy as np
from portraitfx.core import composite
from portraitfx.native import composite_cpp

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/native_benchmark.json')
    p.add_argument('--iterations',type=int,default=60)
    a=p.parse_args()
    if a.iterations<10: p.error('at least 10 iterations required')
    rng=np.random.default_rng(42); rows=[]
    for h,w in [(320,320),(854,480),(720,1280)]:
        for layout in ['contiguous_float32','strided_float32']:
            fg=rng.random((h,w,3),dtype=np.float32)
            bg=rng.random((h,w,3),dtype=np.float32)
            alpha=rng.random((h,w),dtype=np.float32)
            if layout.startswith('strided'): fg=fg[:,:,::-1]; bg=bg[:,:,::-1]
            reference=composite(fg,alpha,bg); result=composite_cpp(fg,alpha,bg)
            error=float(np.max(abs(reference-result)))
            np.testing.assert_allclose(result,reference,rtol=0,atol=2e-7)
            samples={'numpy':[],'cpp':[]}
            for _ in range(5): composite(fg,alpha,bg); composite_cpp(fg,alpha,bg)
            for i in range(a.iterations):
                order=[('numpy',composite),('cpp',composite_cpp)]
                if i%2: order.reverse()
                for name,fn in order:
                    start=time.perf_counter_ns(); fn(fg,alpha,bg)
                    samples[name].append((time.perf_counter_ns()-start)/1e6)
            rows.append(dict(resolution=[w,h],layout=layout,max_abs_error=error,
                samples_ms=samples,timing_ms={k:dict(p50=float(np.median(v)),p95=float(np.percentile(v,95))) for k,v in samples.items()}))
    report=dict(platform=platform.platform(),numpy=np.__version__,iterations=a.iterations,
        scope='Blend only. Same float32 inputs for both backends; includes wrapper conversion, allocation and ctypes. Not end-to-end speed.',
        results=rows)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
