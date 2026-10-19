"""Manifest paths are relative to the manifest; split groups must be disjoint."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np


def load_manifest(path):
    path = Path(path).resolve()
    doc = json.loads(path.read_text())
    rows = doc['samples']
    if not rows: raise ValueError('empty manifest')
    groups, contents = {}, {}
    for row in rows:
        if row['split'] not in ('train', 'val', 'test'): raise ValueError('invalid split')
        if not row.get('group') or not row.get('license'): raise ValueError('group and license required')
        old = groups.setdefault(row['group'], row['split'])
        if old != row['split']: raise ValueError('subject/sequence group crosses splits')
        for key in ('image', 'alpha'):
            p = (path.parent / row[key]).resolve()
            if not p.is_file(): raise ValueError(f'missing {key}: {p}')
            row[key] = str(p)
        digest = hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()
        old = contents.setdefault(digest, row['split'])
        if old != row['split']: raise ValueError('duplicate image crosses splits')
    return doc


def read_pair(row, size=None):
    image = cv2.imread(row['image'])
    alpha = cv2.imread(row['alpha'], cv2.IMREAD_UNCHANGED)
    if image is None or alpha is None or alpha.ndim != 2 or alpha.dtype not in (np.uint8,np.uint16):
        raise ValueError('RGB image and uint8/uint16 grayscale alpha required')
    if image.shape[:2] != alpha.shape: raise ValueError('image/alpha size mismatch')
    alpha = alpha.astype(np.float32) / np.iinfo(alpha.dtype).max
    if size:
        image = cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha,(size,size),interpolation=cv2.INTER_LINEAR)
    return image, alpha
