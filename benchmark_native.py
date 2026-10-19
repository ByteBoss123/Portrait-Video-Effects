"""Synthetic compositing-only benchmark including copies and allocation."""
import json, platform, time
from pathlib import Path
import numpy as np
from portraitfx.core import composite
from portraitfx.native import composite_cpp
rng = np.random.default_rng(42)
rows = []
for height, width in [(320,320),(854,480),(720,1280)]:
    fg = rng.random((height,width,3), dtype=np.float32)[:,:,::-1]
    alpha = rng.random((height,width), dtype=np.float32)
    bg = rng.random((height,width,3))
    reference = composite(fg,alpha,bg)
    result = composite_cpp(fg,alpha,bg)
    error = float(np.max(np.abs(reference-result)))
    np.testing.assert_allclose(result, reference, rtol=0, atol=2e-7)
    samples = {'numpy': [], 'cpp': []}
    for _ in range(5):
        composite(fg,alpha,bg); composite_cpp(fg,alpha,bg)
    for i in range(60):
        order = [('numpy',composite),('cpp',composite_cpp)]
        if i % 2: order.reverse()
        for name,fn in order:
            start = time.perf_counter_ns(); fn(fg,alpha,bg)
            samples[name].append((time.perf_counter_ns()-start)/1e6)
    rows.append({'resolution':[width,height], 'max_abs_error':error,
        'timing_ms':{name:{'p50':float(np.median(v)), 'p95':float(np.percentile(v,95))} for name,v in samples.items()}})
report = {'platform':platform.platform(),'numpy':np.__version__, 'iterations':60,'seed':42,
    'scope':'Synthetic alpha compositing only; includes conversion, allocation, ctypes. Excludes blur, inference and video IO. Not Windows or iPhone performance.', 'results':rows}
Path('results').mkdir(exist_ok=True)
Path('results/native_benchmark.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
