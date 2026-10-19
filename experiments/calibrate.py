"""Chessboard calibration plus undistortion; keep lens, zoom, resolution fixed."""
import argparse,glob,json
from pathlib import Path
import cv2
import numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--images',required=True,help='quoted glob of chessboard photos')
    p.add_argument('--cols',type=int,default=9); p.add_argument('--rows',type=int,default=6)
    p.add_argument('--output',default='data/calibration.json'); p.add_argument('--undistort',nargs='*',default=[])
    a=p.parse_args()
    if min(a.cols,a.rows)<3: p.error('at least 3 inner corners in each direction')
    grid=np.zeros((a.cols*a.rows,3),np.float32);grid[:,:2]=np.mgrid[0:a.cols,0:a.rows].T.reshape(-1,2)
    objects=[]; corners=[]; size=None
    for name in sorted(glob.glob(a.images)):
        gray=cv2.imread(name,cv2.IMREAD_GRAYSCALE)
        if gray is None: raise ValueError(f'cannot read {name}')
        current=(gray.shape[1],gray.shape[0])
        if size and size!=current: raise ValueError('all photos must share resolution')
        size=current
        ok,pts=cv2.findChessboardCornersSB(gray,(a.cols,a.rows))
        if ok: objects.append(grid.copy());corners.append(pts)
    if len(corners)<10: raise ValueError('need at least 10 detected boards at varied angles/positions')
    rms,K,dist,_,_=cv2.calibrateCamera(objects,corners,size,None,None)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({'K':K.tolist(),'distortion':dist.ravel().tolist(),'width':size[0],'height':size[1],
        'calibration_rms_px':float(rms),'boards':len(corners),
        'note':'Reconstruction input must use undistorted images generated with this K. Calibration RMS is an in-sample residual.'},indent=2))
    for name in a.undistort:
        im=cv2.imread(name)
        if im is None or (im.shape[1],im.shape[0])!=size: raise ValueError('undistortion image dimensions mismatch')
        target=Path(name).with_name(Path(name).stem+'_undistorted.png')
        if target.exists(): raise ValueError(f'refusing overwrite: {target}')
        cv2.imwrite(str(target),cv2.undistort(im,K,dist,None,K));print(target)
    print(out)
if __name__=='__main__': main()
