import numpy as np
import json
import os
from os.path import join as pjoin

# Only for loading annotations
import h5py
from scipy.io import loadmat

from scipy.stats import multivariate_normal
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

try:
    from functools import lru_cache  # Needs at least Python 3.2
except ImportError:
    def lru_cache(fn, *a, **kw): return fn


START_TIMES = [5543, 3607, 27244, 31182, 1, 22402, 18968, 46766]
TRAIN_START, TRAIN_END = 49700, 227540


###############################################################################
# Generic utilities


def scale_shape(shape, factors):
    try:
        len(factors)
    except TypeError:
        # It's a number
        factors = (factors, factors)

    return int(shape[0]*factors[0]), int(shape[1]*factors[1])


def argmax2d_xy(arr, smooth=None):
    if smooth is not None:
        sy, sx = smooth
        arr = cv2.GaussianBlur(arr, (sx,sy), 0, borderType=cv2.BORDER_CONSTANT)

    idx = np.unravel_index(arr.argmax(), arr.shape)
    return np.array([idx[1], idx[0]])


def expected_xy(p, thresh=None, magic_thresh=None):
    if magic_thresh is not None:  # Some weak-ass heuristic
        T = magic_thresh

        # And reduce until we can afford it.
        for _ in range(999):  # Avoids infinite loop.
            thresh = T/np.prod(p.shape)
            if np.sum(thresh <= p):
                break
            T = 1 + (T-1)/2
        else:
            print("Warning: expected_xy ran out of steps")

    if thresh is not None:
        p = np.array(p)
        p[p < thresh] = 0
        p /= np.clip(np.sum(p), 1e-5, 1)  # Avoid NaNs

    return np.sum(np.mgrid[:p.shape[0],:p.shape[1]] * p, axis=(1,2))[::-1]


def softmax(x, T=1):
    x = x - np.max(x)
    eh = np.exp(x/T)
    return eh / np.sum(eh)


def softmin(x, T=1):
    return softmax(-x, T=T)


def entropy(x):
    return -np.sum(x*np.log2(np.clip(x, 1e-14, 1)))


def entropy_avg(x):
    return -np.mean(x*np.log2(np.clip(x, 1e-14, 1)))


def entropy_score(x):
    """ Returns a score between 0 and 1 directly proportional to the entropy.
        It is exactly 0 for (near) uniform distributions, and 1 for single peaks.
    """
    e0 = entropy(softmax(np.zeros_like(x)))
    return (e0 - entropy(x))/e0


def entropy_score_avg(x):
    """ Returns a score between 0 and 1 directly proportional to the entropy.
        It is exactly 0 for (near) uniform distributions, and 1 for single peaks.
    """
    e0 = entropy_avg(np.full_like(x, 1/np.prod(x.shape)))
    x = np.array(x)
    x[x < 2/np.prod(x.shape)] = 0
    x /= np.sum(x)
    return (e0 - entropy_avg(x))/e0


def xent(p, q):
    return -np.sum(p*np.log2(np.clip(q, 1e-14, 1)))


def xent_avg(p, q):
    return -np.mean(p*np.log2(np.clip(q, 1e-14, 1)))


def my_choice(candidates, n):
    return np.random.choice(candidates, n, len(candidates) < n)


def randin(lo, hi):
    return lo + np.random.rand()*(hi-lo)


def ramp(e, e0, v0, e1, v1):
    """
    Return `v0` until `e` reaches `e0`, then linearly interpolate
    to `v1` when `e` reaches `e1` and return `v1` thereafter.

    Copyright (C) 2017 Lucas Beyer - http://lucasb.eyer.be =)
    """
    if e < e0:
        return v0
    elif e < e1:
        return v0 + (v1-v0)*(e-e0)/(e1-e0)
    else:
        return v1


def expdec(e, e0, v0, e1, v1):
    """
    Return `v0` until `e` reaches `e0`, then exponentially decay
    to `v1` when `e` reaches `e1` and return `v1` thereafter.

    Copyright (C) 2017 Lucas Beyer - http://lucasb.eyer.be =)
    """
    if e < e0:
        return v0
    elif e < e1:
        return v0 * (v1/v0)**((e-e0)/(e1-e0))
    else:
        return v1


def sane_listdir(where, ext='', sortkey=None):
    """
    Intended for internal use.
    Like `os.listdir`, but:
        - Doesn't include hidden files,
        - Always returns results in a sorted order (pass `sortkey=int` for numeric sort),
        - Optionally only return entries whose name ends in `ext`.
    """
    return sorted((i for i in os.listdir(where) if not i.startswith('.') and i.endswith(ext)), key=sortkey)


def img2df(img, shape=None):
    """
    Convert raw images into what's needed by DeepFried.
    This means: BGR->RGB, HWC->CHW and [0,255]->[0.0,1.0]

    Note that `shape` is (H,W).
    """
    img = resize_img(img, shape=shape)
    img = np.rollaxis(img, 2, 0)  # HWC to CHW
    img = img.astype(np.float32) / 255.0
    return img


def gauss2d_xy(cov, nstd=2, mean=[0,0]):
    """
    guaranteed to return filter of odd shape which also keeps probabilities as probabilities.
    """
    sx, sy = np.sqrt(cov[0][0]), np.sqrt(cov[1][1])
    dx = max(1, round(nstd*sx + abs(mean[0])))
    dy = max(1, round(nstd*sy + abs(mean[1])))
    x, y = np.mgrid[-dy:dy+1:1, -dx:dx+1:1]  # This formulation ensures odd sizes.
    pos = np.dstack((y, x))
    rv = multivariate_normal(mean, cov)
    filt = rv.pdf(pos)
    return filt/np.sum(filt)  # Make sure it's a probability-preserving


def paste_into_middle_2d(x, out_shape=None, fill_value=0, out=None):
    if out is None:
        out = np.full(out_shape, fill_value, dtype=x.dtype)

    out_shape = out.shape

    if x.shape == out_shape:
        return np.array(x)

    h, w = x.shape
    H, W = out_shape
    dy_out, dx_out = max(0, (H-h)/2), max(0, (W-w)/2)
    dy_in, dx_in = max(0, (h-H)/2), max(0, (w-W)/2)
    #assert 0 <= dy and 0 <= dx,  "Something wrong with shape-fixing! {} = ({}-{})//2 and {} = ({}-{})//2".format(dy, H, h, dx, W, w)

    # Stick to bottom on ambiguity
    #out[int(dy_out+0.5):H-int(dy_out),int(dx_out+0.5):W-int(dx_out)] = x[int(dy_in+0.5):h-int(dy_in),int(dx_in+0.5):w-int(dx_in)]

    # Stick to top on ambiguity
    out[int(dy_out):H-int(dy_out+0.5),int(dx_out):W-int(dx_out+0.5)] = x[int(dy_in):h-int(dy_in+0.5),int(dx_in):w-int(dx_in+0.5)]

    # Doesn't work correctly on perfect ambiguity.
    #out[round(dy_out):H-round(dy_out),round(dx_out):W-round(dx_out)] = x[round(dy_in):h-round(dy_in),round(dx_in):w-round(dx_in)]

    return out


###############################################################################
# Video handling, only with OpenCV

try:
    import cv2


    def resize_img(img, shape=None, interp=None, is_chw=False):
        if shape is None:
            return np.array(img)

        if interp is None:
            interp = cv2.INTER_AREA
        elif interp is 'bicubic':
            interp = cv2.INTER_CUBIC
        else:
            raise NotImplementedError("TODO: Interpolation {} in OpenCV".format(interp))

        if is_chw:
            img = np.rollaxis(img, 0, 3)  # CHW to HWC

        img = cv2.resize(img, (shape[1], shape[0]), interpolation=interp)

        if is_chw:
            img = np.rollaxis(img, 2, 0)  # HWC to CHW

        return img


    def resize_map(img, shape, interp='bicubic'):
        return resize_img(img, shape, interp)


    def imread(fname):
        f = cv2.imread(fname)
        if f is None:
            raise ValueError("Couldn't load file {}".format(fname))
        return f[:,:,::-1]


    def imwrite(fname, img):
        cv2.imwrite(fname, img[:,:,::-1])


    def convolve_edge_same(image, filt):
        # 64F is actually faster than 32?!
        return cv2.filter2D(image, cv2.CV_64F, filt, borderType=cv2.BORDER_REPLICATE)


    def convolve_edge_zeropad(image, filt):
        dx1, dx2 = filt.shape[1]//2, filt.shape[1]//2
        dy1, dy2 = filt.shape[0]//2, filt.shape[0]//2
        x = cv2.copyMakeBorder(image, dy1, dy2, dx1, dx2, cv2.BORDER_CONSTANT)
        x = cv2.filter2D(x, -1, filt)
        return x[dy1:-dy2,dx1:-dx2]


    def video_or_open(video):
        # Because can't access cv2.VideoCapture type (only function exposed)
        if type(video).__name__ == 'VideoCapture':
            return video
        else:
            return cv2.VideoCapture(video)


    def vidframes(video):
        return int(video_or_open(video).get(cv2.CAP_PROP_FRAME_COUNT))


    def itervid(video):
        video = video_or_open(video)

        while True:
            good, img = video.read()

            if not good:
                return

            yield img


    def vid2tensor(video, imgproc=lambda x: x, progress=None):
        video = video_or_open(video)

        T = vidframes(video)
        vid = None

        for t, img in enumerate(itervid(video)):
            img = imgproc(img)

            if vid is None:
                vid = np.empty((T,) + img.shape, img.dtype)

            vid[t] = img

            if progress is not None:
                progress(t, T)

        return vid


    def total_frames(basedir, ext='.MTS', subsample=1):
        T = 0
        for f in sane_listdir(basedir, ext=ext):
            T += vidframes(pjoin(basedir, f))//subsample

        return T


except ImportError:
    import scipy


    def resize_img(img, shape=None, interp='bilinear'):
        if shape is None:
            return np.array(img)

        return scipy.misc.imresize(img, shape, interp=interp, mode='RGB')


    def resize_map(img, shape, interp='bicubic'):
        return scipy.misc.imresize(img, shape, interp=interp, mode='F')


    def imwrite(fname, img):
        scipy.misc.imsave(fname, img)


    def convolve_edge_same(image, filt):
        pad_width = int(filt.shape[1] / 2)
        pad_height = int(filt.shape[0] / 2)
        out_img = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='edge')
        out_img = signal.convolve2d(out_img, filt, mode='valid', boundary='fill', fillvalue=0)
        return out_img


###############################################################################
# Box utils
# Unittest see notebook


def intersect(box1, box2):
    l1, t1, w1, h1 = box1
    l2, t2, w2, h2 = box2

    l3 = max(l1, l2)
    t3 = max(t1, t2)
    return l3, t3, min(l1+w1, l2+w2)-l3, min(t1+h1, t2+h2)-t3


def iou(box1, box2):
    l1, t1, w1, h1 = box1
    l2, t2, w2, h2 = box2

    _, _, wi, hi = intersect(box1, box2)

    # They (practically) don't intersect.
    if wi < 1e-5 or hi < 1e-5:
        return 0.0

    i = wi*hi
    u = w1*h1 + w2*h2 - i
    return i/u


def max_iou(r, others):
    if len(others) == 0:
        return 0
    return max(iou(r, o) for o in others)


def argmax_iou(r, others):
    return np.argmax([iou(r, o) for o in others])


def sample_around(boxes, size, imsize=(1,1), nstd=3):
    H, W = imsize
    h, w = size

    # pick one box
    ml, mt, mw, mh = boxes[np.random.choice(len(boxes))]

    # Go around it but stay in image-space!
    #rand = np.random.randint
    #rand = lambda m, M: m + np.random.rand()*(M-m)
    rand = lambda m, M: np.clip((m+M)/2 + np.random.randn()*(M-m)/(2*nstd), m, M)
    l = rand(max(ml-w, 0), min(ml+mw, W-w))
    t = rand(max(mt-h, 0), min(mt+mh, H-h))
    return l, t, w, h


def sample_lonely(boxes, size, region=(0,0,1,1), thresh=1e-2):
    # NOTE: size is HW whereas boxes and region are LTWH
    # TODO: make smarter?
    H, W = size
    xmin, ymin = region[0], region[1]
    xmax, ymax = region[2] - W, region[3] - H
    x = xmin + (xmax-xmin)*np.random.rand()
    y = ymin + (ymax-ymin)*np.random.rand()
    while thresh < max_iou((x, y, W, H), boxes):
        x = xmin + (xmax-xmin)*np.random.rand()
        y = ymin + (ymax-ymin)*np.random.rand()
    return x, y, W, H


def stick_to_bounds(box, bounds=(0,0,1,1)):
    """
    Sticks the given `box`, which is a `(l, t, w, h)`-tuple to the given bounds
    which are also expressed as `(l, t, w, h)`-tuple.
    """
    if bounds is None:
        return box

    l, t, w, h = box
    bl, bt, bw, bh = bounds

    l += max(bl - l, 0)
    l -= max((l+w) - (bl+bw), 0)

    t += max(bt - t, 0)
    t -= max((t+h) - (bt+bh), 0)

    return l, t, w, h


def box_centered(cx, cy, h, w, bounds=(0, 0, 1, 1)):
    """
    Returns a box of size `(h,w)` centered around `(cy,cx)`, but sticked to `bounds`.
    """
    return stick_to_bounds((cx - w / 2, cy - h / 2, w, h), bounds)


def rebox_centered(box, h, w, bounds=(0,0,1,1)):
    """
    Returns a new box of size `(h,w)` centered around the same center as the
    given `box`, which is a `(l,t,w,h)`-tuple, and sticked to `bounds`.
    """
    # box is l t w h
    # size is h w
    l, t, bw, bh = box
    cy, cx = t + bh/2, l + bw/2
    return stick_to_bounds((cx - w/2, cy - h/2, w, h), bounds)


def wiggle_box(box, pct_move=None, factor_size=None):
    """
    As usual, box is `(l,t,w,h)`.
    `pct_move` is max change, in % of boxsize: 0 means no change allowed, 1 means full box-size change allowed.
    `factor_size` is maximum factor of up/down scaling, so 2 means up to half/double bbox-size.
    """
    l, t, w, h = box

    if pct_move is not None:
        try:
            len(pct_move)
        except TypeError:
            pct_move = (pct_move, pct_move)
        l += randin(-w*pct_move[1], w*pct_move[1])
        t += randin(-h*pct_move[0], h*pct_move[0])

    nw, nh = w, h
    if factor_size is not None:
        sf = randin(1, factor_size)
        if np.random.rand() < 0.5:
            sf = 1/sf
        nw *= sf
        nh *= sf

    return rebox_centered((l, t, w, h), nh, nw)


def cutout_rel_chw(img, box):
    """
    Returns a cut-out of `img` (which is CHW) at the *relative* `box` location.
    `box` is a `(l,t,w,h)`-tuple as usual, but in [0,1]-coordinates relative to
    the image size.
    """
    _, H, W = img.shape
    l, t, w, h = box
    return img[:,int(t*H):int(t*H)+int(h*H)
                ,int(l*W):int(l*W)+int(w*W)]


def cutout_abs_hwc(img, box):
    """
    Returns a cut-out of `img` (which is HWC) at the *absolute* `box` location.
    `box` is a `(l,t,w,h)`-tuple as usual, in absolute coordinates.
    """
    l, t, w, h = map(int, box)
    return img[t:t+h,l:l+w]


def box_rel2abs(box,h=1080,w=1920):
    return [box[0]*w, box[1]*h, box[2]*w, box[3]*h]


def box_center_xy(box):
    return [box[0] + box[2] / 2., box[1] + box[3] / 2.]


def ltrb_to_box(ltrb):
    l, t, r, b = ltrb[0], ltrb[1], ltrb[2], ltrb[3]
    return [l, t, r-l, b-t]

###############################################################################
# Frame-switching


def loc2glob(loc, cam):
    # Compute global frame numbers once.
    offset = START_TIMES[cam-1] - 1
    return loc + offset

def glob2loc(glob, cam):
    # Compute global frame numbers once.
    offset = START_TIMES[cam-1] - 1
    return glob - offset

assert loc2glob(1, 1) == 5543
assert glob2loc(loc2glob(2,1),1) == 2



###############################################################################
# Data-handling


def slice_all(f, s):
    return {k: v[s] for k,v in f.items()}


def load_trainval(fname, time_range=[TRAIN_START, TRAIN_END]):
    try:
        m = loadmat(fname)['trainData']
    except NotImplementedError:
        with h5py.File(fname, 'r') as f:
            m = np.array(f['trainData']).T

    data = {
        'Cams': np.array(m[:,0], dtype=int),
        'TIDs': np.array(m[:,1], dtype=int),
        'LFIDs': np.array(m[:,2], dtype=int),
        'boxes': np.array(m[:,3:7], dtype=float),
        'world': np.array(m[:,7:9]),
        'feet': np.array(m[:,9:]),
    }

    # boxes are l t w h
    data['boxes'][:,0] /= 1920
    data['boxes'][:,1] /= 1080
    data['boxes'][:,2] /= 1920
    data['boxes'][:,3] /= 1080

    # Compute global frame numbers once.
    data['GFIDs'] = np.array(data['LFIDs'])
    for icam, t0 in zip(range(1,9), START_TIMES):
        data['GFIDs'][data['Cams'] == icam] += t0 - 1

    #return data
    return slice_all(data, (time_range[0] <= data['GFIDs']) & (data['GFIDs'] <= time_range[1]))


def load_dat(basename):
    desc = json.load(open(basename + '.json', 'r'))
    dtype, shape = desc['dtype'], tuple(desc['shape'])
    Xm = np.memmap(basename, mode='r', dtype=dtype, shape=shape)
    Xa = np.ndarray.__new__(np.ndarray, dtype=dtype, shape=shape, buffer=Xm)
    return Xa

###############################################################################
# Plotting
def get_transparent_colormap(cmap=plt.cm.inferno):
    out_cmap = cmap(np.arange(cmap.N))
    out_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    out_cmap = ListedColormap(out_cmap)
    return out_cmap
