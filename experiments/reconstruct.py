"""Calibrated two-view sparse reconstruction. Not SLAM or dense reconstruction."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np

def reconstruct(p1,p2,K):
    p1,p2,K=(np.asarray(x,dtype=np.float64) for x in (p1,p2,K))
    if p1.shape!=p2.shape or p1.ndim!=2 or p1.shape[1]!=2 or len(p1)<8: raise ValueError('at least 8 matching 2D points required')
    if K.shape!=(3,3) or not np.isfinite(K).all() or min(K[0,0],K[1,1])<=0: raise ValueError('valid 3x3 intrinsic matrix required')
    if not np.isfinite(p1).all() or not np.isfinite(p2).all(): raise ValueError('nonfinite points')
    cv2.setRNGSeed(42)
    E,mask=cv2.findEssentialMat(p1,p2,K,method=cv2.RANSAC,prob=.999,threshold=1.)
    if E is None or E.shape!=(3,3): raise ValueError('essential matrix estimation failed')
    _,R,t,mask=cv2.recoverPose(E,p1,p2,K,mask=mask)
    keep=mask.ravel()!=0; u,v=p1[keep],p2[keep]
    if len(u)<8: raise ValueError('insufficient pose inliers; use translated views of a textured nonplanar scene')
    P1=K@np.column_stack((np.eye(3),np.zeros(3))); P2=K@np.column_stack((R,t))
    h=cv2.triangulatePoints(P1,P2,u.T,v.T)
    valid=np.abs(h[3])>1e-10; h=h[:,valid]; u=u[valid]; v=v[valid]
    X=(h[:3]/h[3]).T; X2=(R@X.T+t).T
    def project(P,X):
        q=(P@np.column_stack((X,np.ones(len(X)))).T).T
        return q[:,:2]/q[:,2,None]
    errors=(np.linalg.norm(project(P1,X)-u,axis=1)+np.linalg.norm(project(P2,X)-v,axis=1))/2
    c2=(-R.T@t).ravel(); ray1=X/np.linalg.norm(X,axis=1)[:,None]
    ray2=(X-c2)/np.linalg.norm(X-c2,axis=1)[:,None]
    angles=np.degrees(np.arccos(np.clip(np.sum(ray1*ray2,axis=1),-1,1)))
    good=(X[:,2]>0)&(X2[:,2]>0)&np.isfinite(X).all(axis=1)&(errors<2)&(angles>.5)
    if good.sum()<8: raise ValueError('insufficient stable triangulation; low parallax or poor matches')
    return X[good], {'input_matches':len(p1),'pose_inliers':int(keep.sum()),'points':int(good.sum()),
        'mean_reprojection_error_px':float(errors[good].mean()),'median_parallax_deg':float(np.median(angles[good])),
        'R_world_to_camera2':R.tolist(),'t_world_to_camera2_unit_scale':t.ravel().tolist(),
        'camera_centers':[[0.,0.,0.],c2.tolist()],
        'notes':['Two views only; scale is arbitrary.','Reprojection error uses fitting correspondences, not independent ground truth.','No bundle adjustment, SLAM, neural rendering or dense surface reconstruction.']}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--demo',action='store_true'); p.add_argument('--left'); p.add_argument('--right')
    p.add_argument('--calibration',help='JSON with K, width, height; photos must be undistorted and share intrinsics')
    p.add_argument('--output',default='results/reconstruction'); a=p.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    if a.demo:
        rng=np.random.default_rng(42); truth=rng.uniform([-1,-.7,4],[1,.7,8],(300,3))
        K=np.array([[700,0,320],[0,700,240],[0,0,1.]])
        R,_=cv2.Rodrigues(np.array([0.,.03,0.])); t=np.array([.7,0.,0.])
        def proj(X):
            q=(K@X.T).T; return q[:,:2]/q[:,2,None]
        p1=proj(truth); p2=proj((R@truth.T).T+t)
    else:
        if not all([a.left,a.right,a.calibration]): p.error('supply --demo or --left --right --calibration')
        left,right=cv2.imread(a.left),cv2.imread(a.right)
        if left is None or right is None or left.shape!=right.shape: raise ValueError('readable matching-size images required')
        cal=json.loads(Path(a.calibration).read_text()); K=np.array(cal['K'])
        if [left.shape[1],left.shape[0]]!=[cal['width'],cal['height']]: raise ValueError('calibration dimensions mismatch')
        sift=cv2.SIFT_create(nfeatures=6000)
        k1,d1=sift.detectAndCompute(cv2.cvtColor(left,cv2.COLOR_BGR2GRAY),None)
        k2,d2=sift.detectAndCompute(cv2.cvtColor(right,cv2.COLOR_BGR2GRAY),None)
        if d1 is None or d2 is None: raise ValueError('not enough texture')
        matches=[pair[0] for pair in cv2.BFMatcher().knnMatch(d1,d2,k=2) if len(pair)==2 and pair[0].distance<.7*pair[1].distance]
        # One-to-one correspondences prevent repeated target features dominating geometry.
        unique={}
        for m in sorted(matches,key=lambda m:m.distance): unique.setdefault(m.trainIdx,m)
        matches=list(unique.values()); p1=np.array([k1[m.queryIdx].pt for m in matches]); p2=np.array([k2[m.trainIdx].pt for m in matches])
        cv2.imwrite(str(out/'matches.jpg'),cv2.drawMatches(left,k1,right,k2,matches[:100],None))
    points,report=reconstruct(p1,p2,K); report['synthetic']=a.demo
    if a.demo:
        r=np.array(report['R_world_to_camera2']); tt=np.array(report['t_world_to_camera2_unit_scale'])
        report['rotation_error_deg']=float(np.degrees(np.arccos(np.clip((np.trace(r@R.T)-1)/2,-1,1))))
        report['translation_direction_error_deg']=float(np.degrees(np.arccos(np.clip(tt@(t/np.linalg.norm(t)),-1,1))))
    with (out/'points.ply').open('w') as f:
        f.write(f'ply\nformat ascii 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
        np.savetxt(f,points)
    (out/'report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
