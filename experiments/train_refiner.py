"""Train a residual alpha refiner on frozen RVM predictions and labeled images.
This is supervised adaptation, not fine-tuning RVM weights or video training.
"""
import argparse, hashlib, json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from .data import load_manifest, read_pair
from portraitfx.core import RVM, alpha_metrics

class Refiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(4,16,3,padding=1),nn.ReLU(),
                               nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,1,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):
        return (x[:,3:4]+.25*torch.tanh(self.net(x))).clamp(0,1)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True);p.add_argument('--model',required=True)
    p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--size',type=int,default=128);p.add_argument('--batch',type=int,default=4)
    p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--allow-synthetic',action='store_true')
    a=p.parse_args()
    if min(a.epochs,a.size,a.batch)<1: p.error('positive epochs, size and batch required')
    doc=load_manifest(a.manifest)
    if doc.get('synthetic',False) and not a.allow_synthetic:
        p.error('real labeled data required; --allow-synthetic is only a software check')
    splits={s:[r for r in doc['samples'] if r['split']==s] for s in ('train','val','test')}
    if any(not v for v in splits.values()):p.error('nonempty train/val/test required')
    out=Path(a.output)
    if out.exists() and any(out.iterdir()):p.error('output must be empty')
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(a.seed);torch.set_num_threads(2);cv2.setNumThreads(2)
    torch.use_deterministic_algorithms(True)
    rvm=RVM(a.model)
    def sample(row):
        # Stream pairs rather than retaining the entire dataset on GPU.
        im,truth=read_pair(row,a.size)
        _,base=rvm.predict(im,independent=True)
        x=np.concatenate([im[:,:,::-1].astype(np.float32)/255,base[...,None]],axis=2)
        return torch.from_numpy(x.transpose(2,0,1).copy()),torch.from_numpy(truth[None]),base
    model=Refiner().to(a.device)
    optim=torch.optim.Adam(model.parameters(),lr=1e-3)
    def evaluate(rows):
        model.eval(); result=[]
        with torch.no_grad():
            for row in rows:
                x,y,base=sample(row)
                pred=model(x[None].to(a.device))[0,0].cpu().numpy()
                result.append(dict(group=row['group'],baseline=alpha_metrics(base,y[0].numpy()),
                                   refined=alpha_metrics(pred,y[0].numpy())))
        groups=sorted({r['group'] for r in result})
        summary={}
        for mode in ('baseline','refined'):
            summary[mode]={}
            for metric in ('mae','mse','boundary_band_mae'):
                means=[]
                for group in groups:
                    vals=[r[mode][metric] for r in result if r['group']==group and r[mode][metric] is not None]
                    if vals:means.append(float(np.mean(vals)))
                summary[mode][metric]=float(np.mean(means)) if means else None
        return dict(group_macro=summary,per_image=result)
    history=[]; best=float('inf')
    for epoch in range(a.epochs):
        model.train(); losses=[]
        order=torch.randperm(len(splits['train'])).tolist()
        for start in range(0,len(order),a.batch):
            pairs=[sample(splits['train'][i]) for i in order[start:start+a.batch]]
            x=torch.stack([v[0] for v in pairs]).to(a.device)
            y=torch.stack([v[1] for v in pairs]).to(a.device)
            pred=model(x)
            loss=(pred-y).abs().mean()
            optim.zero_grad();loss.backward();optim.step();losses.append(loss.item())
        val=evaluate(splits['val'])
        score=val['group_macro']['refined']['mae']
        history.append(dict(epoch=epoch+1,train_l1=float(np.mean(losses)),val_group_mae=score))
        print(history[-1],flush=True)
        if score<best:
            best=score
            torch.save(dict(state_dict=model.state_dict(),size=a.size),out/'best.pt')
    model.load_state_dict(torch.load(out/'best.pt',map_location=a.device,weights_only=True)['state_dict'])
    test=evaluate(splits['test'])
    report=dict(task='Residual alpha refiner trained on frozen independent-frame RVM; not RVM fine-tuning',
        synthetic=bool(doc.get('synthetic',False)),config=vars(a),history=history,
        test=test,split_counts={k:len(v) for k,v in splits.items()},
        manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
        rvm_sha256=hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
        note='Validation selects checkpoint; test evaluated once afterward. Square resize changes aspect ratio. No temporal training or generalization claim.')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    model.eval()
    torch.jit.trace(model.cpu(),torch.zeros(1,4,a.size,a.size)).save(str(out/'refiner.pt'))
    print(json.dumps(test['group_macro'],indent=2))
if __name__=='__main__':main()
