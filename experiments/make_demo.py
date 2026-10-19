"""Generate original synthetic shapes ONLY for testing code, not CV claims."""
import argparse, json
from pathlib import Path
import cv2
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='data/synthetic'); a=p.parse_args()
    root=Path(a.output); root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(42); rows=[]
    for split,n in [('train',32),('val',8),('test',8)]:
        for i in range(n):
            alpha=np.zeros((64,64),np.uint8)
            center=tuple(int(v) for v in rng.integers(16,48,2))
            cv2.circle(alpha,center,int(rng.integers(7,18)),255,-1)
            alpha=cv2.GaussianBlur(alpha,(5,5),1)
            bg=rng.integers(5,70,(64,64,3)).astype(np.float32)
            fg=rng.integers(160,255,3).astype(np.float32)
            image=(fg*alpha[...,None]/255+bg*(1-alpha[...,None]/255)).astype(np.uint8)
            name=f'{split}_{i:03d}'
            cv2.imwrite(str(root/f'{name}.png'),image)
            cv2.imwrite(str(root/f'{name}_alpha.png'),alpha)
            rows.append(dict(image=f'{name}.png',alpha=f'{name}_alpha.png',split=split,
                             group=name,frame=0,license='Original synthetic fixture; CC0'))
    (root/'manifest.json').write_text(json.dumps(dict(synthetic=True,samples=rows),indent=2))
    print(root/'manifest.json')
if __name__=='__main__': main()
