"""Download a pinned, bounded P3M-10K subset with original embedded masks.
Requires requests, pyarrow, Pillow. See REAL_DATA.md for provenance and terms.
"""
import argparse
import hashlib
import io
import json
import random
from pathlib import Path
import requests
import pyarrow.parquet as pq
from PIL import Image, ImageChops

REVISION = 'a239de9297bd56e372cf3c9c82f7085d59e71219'
FILES = {
    'train-00000-of-00011.parquet': '0d749a19840ce093930248ad51d89d8262f3fba9dfd88765425853081eec4f4a',
    'P3M_500_P-00000-of-00001.parquet': '3d07f22ec6c4763f7900fbc46f485f32cb3f5c43be326e54bd913b55b6f83c64',
}
BASE = f'https://huggingface.co/datasets/nobg/P3M-10K/resolve/{REVISION}/data/'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='data/p3m-real')
    p.add_argument('--cache', default='data/p3m-cache')
    a = p.parse_args()
    out, cache = Path(a.output), Path(a.cache)
    out.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists(): p.error('manifest already exists; use a new output directory')
    samples = []
    for name, expected in FILES.items():
        path = cache/name
        if not path.exists():
            partial = path.with_suffix('.partial')
            with requests.get(BASE+name, stream=True, timeout=120) as response:
                response.raise_for_status()
                with partial.open('wb') as f:
                    for chunk in response.iter_content(1024*1024): f.write(chunk)
            partial.replace(path)
        if digest(path) != expected: raise ValueError(f'checksum mismatch: {path}')
        parquet = pq.ParquetFile(path)
        indices = list(range(parquet.metadata.num_rows))
        random.Random(42).shuffle(indices)
        # Predeclared image-level random subset; official test is never used for selection.
        selected = ({i:'train' for i in indices[:128]} | {i:'val' for i in indices[128:160]}
                    if name.startswith('train') else {i:'test' for i in indices[:64]})
        row_index = 0
        for batch in parquet.iter_batches(batch_size=16, columns=['image','mask']):
            for row in batch.to_pylist():
                i = row_index; row_index += 1
                if i not in selected: continue
                stem = ('train' if name.startswith('train') else 'testP') + f'_{i:05d}'
                folder = out/selected[i]; folder.mkdir(exist_ok=True)
                im, mask = Image.open(io.BytesIO(row['image']['bytes'])), Image.open(io.BytesIO(row['mask']['bytes']))
                if mask.format not in ('PNG','TIFF'): raise ValueError(f'lossy alpha source: {mask.format}')
                if im.size != mask.size: raise ValueError('image/mask mismatch')
                im.load()
                ext = {'JPEG':'.jpg','PNG':'.png'}.get(im.format)
                if ext is None: raise ValueError(f'unexpected image format: {im.format}')
                ip, ap = folder/(stem+ext), folder/(stem+'_alpha.png')
                ip.write_bytes(row['image']['bytes'])
                if mask.mode == 'RGB':
                    r,g,b=mask.split()
                    if ImageChops.difference(r,g).getbbox() or ImageChops.difference(r,b).getbbox():
                        raise ValueError('RGB alpha channels differ')
                    mask=r
                if mask.mode != 'L': raise ValueError(f'unexpected alpha mode: {mask.mode}')
                encoded=io.BytesIO(); mask.save(encoded, format='PNG')
                ap.write_bytes(encoded.getvalue())
                with Image.open(ip) as check: check.load()
                with Image.open(ap) as check: check.load()
                samples.append(dict(image=ip.relative_to(out).as_posix(), alpha=ap.relative_to(out).as_posix(),
                    split=selected[i], group=stem, license='P3M-10K MIT; original dataset release agreement applies',
                    source_file=name, source_row=i, image_sha256=digest(ip), alpha_sha256=digest(ap)))
        print(name, 'extracted', len(selected), flush=True)
    doc = dict(dataset='P3M-10K bounded face-blurred subset', synthetic=False,
        mirror='nobg/P3M-10K', revision=REVISION, source_files=FILES, seed=42,
        grouping='image-level only; subject identities unavailable; no identity-disjoint claim',
        samples=sorted(samples, key=lambda r:(r['split'], r['group'])))
    (out/'manifest.json').write_text(json.dumps(doc, indent=2))
    print(out/'manifest.json')

if __name__ == '__main__': main()
