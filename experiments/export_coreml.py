"""Run on a Mac with coremltools; exported model must still be device-validated."""
import argparse
from pathlib import Path
import torch
from torch import nn
from .train import AlphaNet

class ImageOutput(nn.Module):
    def __init__(self,model): super().__init__(); self.model=model
    def forward(self,x): return self.model(x)*255.

def main():
    import coremltools as ct
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--output',default='AlphaNet.mlpackage'); a=p.parse_args()
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    model=AlphaNet(); model.load_state_dict(ckpt['state_dict']); model.eval(); size=ckpt['size']
    wrapped=ImageOutput(model).eval(); shape=(1,3,size,size)
    trace=torch.jit.trace(wrapped,torch.zeros(shape))
    converted=ct.convert(trace,convert_to='mlprogram',minimum_deployment_target=ct.target.iOS15,
        inputs=[ct.ImageType(name='image',shape=shape,scale=1/255.,color_layout=ct.colorlayout.RGB)],
        outputs=[ct.ImageType(name='alpha',color_layout=ct.colorlayout.GRAYSCALE)])
    converted.short_description='Small experimental RGB-to-alpha CNN. Check training provenance before use.'
    converted.save(a.output); print(Path(a.output).resolve())
if __name__=='__main__': main()
