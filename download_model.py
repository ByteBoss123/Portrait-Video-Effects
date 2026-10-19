"""Download original RVM model; do not redistribute without its license."""
from pathlib import Path
import urllib.request,hashlib,json
url='https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx'
p=Path('models/rvm_mobilenetv3_fp32.onnx'); p.parent.mkdir(exist_ok=True)
if not p.exists():
    tmp=p.with_suffix('.tmp'); urllib.request.urlretrieve(url,tmp); tmp.replace(p)
print(json.dumps({'path':str(p),'source':url,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2))
