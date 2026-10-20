import cv2
import numpy as np
import onnxruntime as ort

class RVM:
    def __init__(self, model, ratio=.5, threads=2):
        if not 0 < ratio <= 1:
            raise ValueError('ratio must be in (0,1]')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
        self.dtype = np.float16 if self.session.get_inputs()[0].type == 'tensor(float16)' else np.float32
        self.ratio = np.array([ratio], np.float32)
        self.reset()

    def reset(self):
        self.states = [np.zeros((1,1,1,1), self.dtype) for _ in range(4)]
        self.shape = None

    def predict(self, bgr, independent=False):
        if independent or self.shape != bgr.shape:
            self.reset()
        self.shape = bgr.shape
        src = np.ascontiguousarray(bgr[:,:,::-1].transpose(2,0,1)[None], dtype=self.dtype) / 255
        feeds = dict(src=src, downsample_ratio=self.ratio)
        feeds.update({f'r{i+1}i': value for i,value in enumerate(self.states)})
        fgr, alpha, *self.states = self.session.run(['fgr','pha','r1o','r2o','r3o','r4o'], feeds)
        return np.clip(fgr[0].transpose(1,2,0)[:,:,::-1],0,1), np.clip(alpha[0,0],0,1)

def composite(foreground, alpha, background, strength=1.0):
    if not np.isfinite(strength) or not 0 <= strength <= 1: raise ValueError("strength must be in [0,1]")
    if foreground.shape != background.shape or alpha.shape != foreground.shape[:2]:
        raise ValueError('foreground, alpha and background dimensions must match')
    a = (1-strength*(1-np.clip(alpha,0,1)))[...,None]
    return np.clip(foreground*a + background*(1-a),0,1)

def background_blur(frame, alpha, sigma=12, scale=1.0):
    if not 0 < scale <= 1 or sigma <= 0:
        raise ValueError('blur scale must be in (0,1] and sigma positive')
    if scale < 1:
        h,w = alpha.shape
        size = (max(2,round(w*scale)),max(2,round(h*scale)))
        small = cv2.resize(frame,size,interpolation=cv2.INTER_AREA)
        matte = cv2.resize(alpha,size,interpolation=cv2.INTER_AREA)
        blurred = background_blur(small,matte,max(.1,sigma*scale))
        return cv2.resize(blurred,(w,h),interpolation=cv2.INTER_LINEAR)
    # Normalized masked convolution reduces foreground color bleeding.
    weight = (1-alpha).astype(np.float32)
    den = cv2.GaussianBlur(weight,(0,0),sigma)
    num = cv2.GaussianBlur(frame.astype(np.float32)/255*weight[...,None],(0,0),sigma)
    return np.where(den[...,None] > 1e-5, num/np.maximum(den[...,None],1e-5), frame.astype(np.float32)/255)

def alpha_metrics(pred, truth):
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError('matching 2-D alpha mattes required')
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError('nonfinite alpha values')
    if min(pred.min(),truth.min()) < 0 or max(pred.max(),truth.max()) > 1:
        raise ValueError('alpha must be in [0,1]')
    d = pred.astype(float)-truth.astype(float)
    band = cv2.dilate((truth>.5).astype('uint8'),np.ones((5,5),np.uint8))-cv2.erode((truth>.5).astype('uint8'),np.ones((5,5),np.uint8))
    return dict(mae=float(abs(d).mean()),mse=float((d*d).mean()),boundary_band_mae=float(abs(d)[band>0].mean()) if band.any() else None)
