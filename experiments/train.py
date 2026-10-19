"""Train a small RGB-to-alpha CNN from scratch. Not RVM fine-tuning."""
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .data import load_manifest, read_pair

class AlphaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,1,1),nn.Sigmoid())
    def forward(self,x): return self.net(x)

def tensors(rows,size):
    xs,ys=[],[]
    for r in rows:
        im,alpha=read_pair(r,size)
        xs.append(im[:,:,::-1].transpose(2,0,1).copy()/255.)
        ys.append(alpha[None])
    return torch.tensor(np.array(xs),dtype=torch.float32),torch.tensor(np.array(ys),dtype=torch.float32)

def scores(model,x,y):
    with torch.no_grad():
        pred=model(x)
        return {'mae':float((pred-y).abs().mean()),'mse':float(((pred-y)**2).mean())}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True)
    p.add_argument('--output',default='results/training'); p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--size',type=int,default=64); p.add_argument('--batch',type=int,default=8)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='cpu')
    a=p.parse_args()
    if min(a.epochs,a.size,a.batch)<1: p.error('epochs, size and batch must be positive')
    out=Path(a.output)
    if out.exists() and any(out.iterdir()): raise ValueError('choose an empty output folder to preserve experiments')
    out.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    doc=load_manifest(a.manifest)
    splits={s:[r for r in doc['samples'] if r['split']==s] for s in ['train','val','test']}
    if any(not v for v in splits.values()): raise ValueError('nonempty train/val/test required')
    # Small-data baseline: resize and load into RAM. Use a streaming Dataset for large collections.
    data={s:tuple(t.to(a.device) for t in tensors(rows,a.size)) for s,rows in splits.items()}
    model=AlphaNet().to(a.device); optim=torch.optim.Adam(model.parameters(),lr=.003)
    baseline=scores(model,*data['test'])
    best=float('inf'); history=[]
    x,y=data['train']
    for epoch in range(a.epochs):
        model.train(); losses=[]
        for ids in torch.randperm(len(x),device=a.device).split(a.batch):
            optim.zero_grad(); loss=((model(x[ids])-y[ids])**2).mean(); loss.backward(); optim.step()
            losses.append(float(loss.detach()))
        model.eval(); val=scores(model,*data['val'])
        history.append({'epoch':epoch+1,'train_mse':float(np.mean(losses)),'val_mae':val['mae']})
        if val['mae']<best:
            best=val['mae']; torch.save({'state_dict':model.cpu().state_dict(),'size':a.size},out/'best.pt'); model.to(a.device)
        print(history[-1],flush=True)
    ckpt=torch.load(out/'best.pt',map_location=a.device,weights_only=True); model.load_state_dict(ckpt['state_dict'])
    model.eval(); final=scores(model,*data['test'])
    constant=float(data['train'][1].mean())
    report={'synthetic':bool(doc.get('synthetic',False)),'task':'small RGB-to-alpha CNN trained from scratch, not RVM',
        'seed':a.seed,'config':vars(a),'split_counts':{s:len(v) for s,v in splits.items()},'history':history,
        'untrained_test':baseline,'train_mean_baseline_test_mae':float((data['test'][1]-constant).abs().mean()),
        'selected_model_test':final,'manifest_sha256':hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
        'torch_version':str(torch.__version__),'note':'Test metrics do not select checkpoints. Synthetic results are software checks, not person-matting quality.'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    # Save a traced artifact for portable inference and later Core ML conversion.
    torch.jit.trace(model.cpu(),torch.zeros(1,3,a.size,a.size)).save(str(out/'alpha_net.pt'))
    print(json.dumps(final))
if __name__=='__main__': main()
