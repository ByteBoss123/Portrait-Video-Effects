"""Sequence-balanced RVM recurrence ablation on a labeled manifest."""
import argparse,json,time,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from portraitfx.core import RVM, alpha_metrics
from .data import load_manifest,read_pair

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--model',required=True)
    p.add_argument('--split',choices=['val','test'],default='test'); p.add_argument('--ratio',type=float,default=.5)
    p.add_argument('--output',default='results/video_quality.json'); a=p.parse_args()
    doc=load_manifest(a.manifest); groups=defaultdict(list)
    for r in doc['samples']:
        if r['split']==a.split: groups[(r['group'],r.get('sequence',r['group']))].append(r)
    if not groups: raise ValueError('no samples in requested split')
    model=RVM(a.model,a.ratio); results={}
    for independent in [False,True]:
        sequences=[]
        for name,rows in sorted(groups.items()):
            rows=sorted(rows,key=lambda r:r['frame'])
            if len({r['frame'] for r in rows})!=len(rows): raise ValueError('duplicate frame index')
            model.reset(); per_frame=[]
            for row in rows:
                im,truth=read_pair(row); t=time.perf_counter()
                _,pred=model.predict(im,independent=independent)
                elapsed=(time.perf_counter()-t)*1000
                per_frame.append({'frame':row['frame'],**alpha_metrics(pred,truth),'inference_ms':elapsed})
            keys=['mae','mse','boundary_band_mae']
            mean={k:float(np.mean([r[k] for r in per_frame if r[k] is not None])) if any(r[k] is not None for r in per_frame) else None for k in keys}
            sequences.append({'group':name[0],'sequence':name[1],'frames':len(rows),'metrics':mean,'per_frame':per_frame})
        results['independent' if independent else 'recurrent']={'sequences':sequences,
            'sequence_macro_mae':float(np.mean([s['metrics']['mae'] for s in sequences]))}
    report={'synthetic':doc.get('synthetic',False),'split':a.split,'ratio':a.ratio,'results':results,
        'model_sha256':hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
        'notes':['Sequence macro averages weight videos equally.','No motion-compensated temporal metric or accuracy threshold claimed.','Cold first-frame timings included; quality is the purpose of this comparison.']}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)); print(out)
if __name__=='__main__': main()
