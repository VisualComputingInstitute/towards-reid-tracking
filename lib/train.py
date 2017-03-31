import numpy as np
import lib


def random_train_lfids(cam, nimg):
    # The first and last frames in trainval annotations!
    return np.random.randint(
        lib.glob2loc(49700, cam),
        lib.glob2loc(227540, cam)+1,  # NOTE: +1 because randing is excluding last.
        size=nimg
    )


def get_batch_iou(Ximgs, Xtr, img_per_cam, boxsize, subsamp, samplefn):
    samples, ious = [], []
    for i in range(len(Ximgs)):
        icam = i+1

        mask_cam = Xtr['Cams'] == icam
        while len(samples[i*img_per_cam:]) < img_per_cam:
            lfid = random_train_lfids(cam=icam, nimg=1)[0]

            # Make it fit to the subsampling factor
            fidx = lfid//subsamp
            lfid = fidx*subsamp

            image = Ximgs[i][fidx]
            mask_boxes = Xtr['LFIDs'] == lfid  # Then bottleneck!
            mask_boxes &= mask_cam
            boxes = Xtr['boxes'][np.where(mask_boxes)[0]]

            # Empty image? Not interesting!
            if not len(boxes):
                continue

            # Now get one box that is near others
            _, H, W = image.shape
            nearby = samplefn(boxes, (boxsize[0]/H, boxsize[1]/W))
            iou = lib.max_iou(nearby, boxes)

            # l, t, w, h
            co = np.array(lib.cutout_rel_chw(image, nearby))
            if co.shape[1:] != boxsize:  # I don't trust myself anymore.
                print("WARNING: Bad cutout for cam {} fidx {} lfid {} cut {}".format(icam, fidx, lfid, nearby))
                continue

            samples.append(co)
            ious.append(iou)

    return np.array(samples, dtype=np.float32)/255.0, np.array(ious, dtype=np.float32)
    # TODO: and one that is way off others.
    # TODO: have the box-size vary and resize it to fixed?


def sfn_overlap(nstd):
    return lambda boxes, size: lib.sample_around(boxes, size, nstd=nstd)


def sfn_lonely(thresh=1e-2):
    return lambda boxes, size: lib.sample_lonely(boxes, size, thresh=thresh)


def get_pk_batch(Ximgs, Xtr, ntid, img_per_tid, boxsize, subsamp, ret_iou=False, **augkw):
    """ For now, `augkw` is only `pct_move` and a boolean `flip`.
    Specifically, for `factor_size`, I'd have to add a scaling call, which I don't want yet.
    """
    all_tids = np.unique(Xtr['TIDs'])  #50ms, 215ms if return_index=True

    # Select some people to use.
    tids = np.random.choice(all_tids, ntid, replace=False)

    cutouts, ious = [], []
    for tid in tids:
        idxs = lib.my_choice(np.where(Xtr['TIDs'] == tid)[0], img_per_tid)

        for boxidx in idxs:
            box = Xtr['boxes'][boxidx]
            cid = Xtr['Cams'][boxidx]
            lfid = Xtr['LFIDs'][boxidx]

            # Make it fit to the subsampling factor
            assert lfid % subsamp == 0, "Not implemented yet!"
            fidx = lfid//subsamp

            image = Ximgs[cid-1][fidx]
            _, H, W = image.shape
            box = lib.rebox_centered(box, h=boxsize[0]/H, w=boxsize[1]/W)
            box = lib.wiggle_box(box, pct_move=augkw.get('pct_move', None))
            co = np.array(lib.cutout_rel_chw(image, lib.stick_to_bounds(box)))
            if augkw.get('flip', False) and np.random.rand() < 0.5:
                co = co[:,:,::-1]
            cutouts.append(co)

            if ret_iou:
                lfid = fidx*subsamp
                mask_boxes = (Xtr['LFIDs'] == lfid) & (Xtr['Cams'] == cid)
                boxes = Xtr['boxes'][np.where(mask_boxes)[0]]
                iou = lib.max_iou(box, boxes)
                ious.append(iou)

    Ximg = np.array(cutouts, dtype=np.float32)/255.0
    ypid = np.repeat(tids, img_per_tid).astype(np.uint32)
    if not ret_iou:
        return Ximg, ypid
    else:
        return Ximg, ypid, np.array(ious, dtype=np.float32)
    # TODO: have the box-size vary and resize it to fixed?


def stat(f, name, e, b, vals):
    if name not in f:
        # NOTE: can't use require_dataset because of shape/growing.
        vals = np.asarray(vals)
        fillval = np.nan if np.issubdtype(vals.dtype, np.float) else 0
        f.create_dataset(name, (e, b) + vals.shape, dtype=vals.dtype,
                         maxshape=(None, None) + vals.shape, fillvalue=fillval)
    ds = f[name]

    if ds.shape[0] <= e:
        ds.resize(e+1, axis=0)
    if ds.shape[1] <= b:
        ds.resize(b+1, axis=1)

    ds[e,b] = vals
    try:
        ds.flush()
    except AttributeError:
        pass  # Only for very modern hdf5 libs, e.g. not on Ubuntu 14


def get_decayparam(opt):
    # This could be better handled in the optimizers themselves.
    lrname = 'alpha' if 'alpha' in opt.hyperparams else 'lr'
    if lrname not in opt.hyperparams:
        raise NotImplementedError("Don't know about learning-rate of this optimizer")
    return opt.hyperparams[lrname], lrname
