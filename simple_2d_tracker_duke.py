#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

import argparse
from os.path import join as pjoin
from os import makedirs
import time, datetime

# the usual suspects
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
#mpl.use('GTK')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.linalg import block_diag, inv
from scipy.io import loadmat

#tracker stuff
import lib
from simple_track_duke import Track
import h5py
from scipy.spatial.distance import euclidean,mahalanobis
from munkres import Munkres, print_matrix
from semifake import SemiFakeNews

SEQ_FPS = 60.0
SEQ_DT = 1./SEQ_FPS
SEQ_SHAPE = (1080, 1920)
STATE_SHAPE = (270, 480)
HOT_CMAP = lib.get_transparent_colormap()
#NUM_CAMS = 2 # which cam to consider (from 1 to NUM_CAMS), max: 8
SCALE_FACTOR = 0.5


g_frames = 0  # Global counter for correct FPS in all cases

def n_active_tracks(tracklist):
    return '{:2d} +{:2d} +{:2d} ={:2d}'.format(
        sum(t.status == 'matched' for t in tracklist),
        sum(t.status == 'missed' for t in tracklist),
        sum(t.status == 'init' for t in tracklist),
        len(tracklist),
    )
    # from collections import Counter
    #return str(Counter(t.status for t in tracklist).most_common())


def shall_vis(args, curr_frame):
    return args.vis and (curr_frame - args.t0) % args.vis == 0


def embed_crops_at(net, image, xys, debug_out_dir=None, debug_cam=None, debug_curr_frame=None):
    H, W, _ = image.shape
    crops = [lib.cutout_abs_hwc(image, lib.box_centered(xy[0]*SCALE_FACTOR, xy[1]*SCALE_FACTOR,
                                                        h=128*2*SCALE_FACTOR, w=48*2*SCALE_FACTOR, bounds=(0, 0, W, H))) for xy in xys]

    if debug_out_dir is not None:
        for icrop, crop in enumerate(crops):
            lib.imwrite(pjoin(debug_out_dir, 'crops', 'cam{}-frame{}-{}.jpg'.format(debug_cam, debug_curr_frame, icrop)), crop)

    return net.embed_crops(crops)


def load_or_reuse(image, args, icam, frame):
    if image is not None:
        return image
    framedir = 'frames-0.5' if SCALE_FACTOR == 0.5 else 'frames'
    return plt.imread(pjoin(args.basedir, framedir, 'camera{}/{}.jpg'.format(icam, lib.glob2loc(frame, icam))))


#@profile
def main(net, args):
    eval_path = pjoin(args.outdir, 'results/run_{:%Y-%m-%d_%H:%M:%S}.txt'.format(datetime.datetime.now()))
    if args.debug:
        debug_dir = pjoin(args.outdir, 'debug/run_{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now()))
        makedirs(pjoin(debug_dir, 'crops'), exist_ok=True)
    else:
        debug_dir = None


    CAMS = args.cams

    track_lists = [[] for _ in range(len(CAMS))]
    already_tracked_gids = [[] for _ in range(len(CAMS))]
    track_id = 1
    det_lists = read_detections(CAMS)
    gt_list = load_trainval(pjoin(args.basedir, 'ground_truth/trainval.mat'),time_range=[127720, 187540]) #train_val_mini
    APP_THRESH = 6 #7 for ReID embeddings, 200 for euclidean pixel distance
    DIST_THRESH = 200  # 7 for ReID embeddings, 200 for euclidean pixel distance
    DET_INIT_THRESH = 0.3
    DET_CONTINUE_THRESH = -0.3
    m = Munkres()

    per_cam_gts = [lib.slice_all(gt_list, gt_list['Cams'] == icam) for icam in CAMS]

    # ===Tracking fun begins: iterate over frames===
    # TODO: global time (duke)
    for curr_frame in range(args.t0, args.t1+1):
        print("\rFrame {}, {} matched/missed/init/total tracks, {} total seen".format(curr_frame, ', '.join(map(n_active_tracks, track_lists)), sum(map(len, track_lists))), end='', flush=True)

        for icam, det_list, gt_list, track_list, already_tracked in zip(CAMS, det_lists, per_cam_gts, track_lists, already_tracked_gids):
            image = None

            curr_dets = det_list[np.where(det_list[:,1] == lib.glob2loc(curr_frame, icam))[0]]
            curr_dets = curr_dets[curr_dets[:,-1] > DET_CONTINUE_THRESH]

            curr_gts = lib.slice_all(gt_list, gt_list['GFIDs'] == curr_frame)


            # ===visualization===
            # First, plot what data we have before doing anything.
            if shall_vis(args, curr_frame):
                fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(20, 12))
                (ax_tl, ax_tr), (ax_bl, ax_br) = axes
                axes = axes.flatten()

                for ax in axes:
                    image = load_or_reuse(image, args, icam, curr_frame)
                    ax.imshow(image, extent=[0, 1920, 1080, 0])

                # plot (active) tracks
                ax_tl.set_title('Groundtruth')
                ax_tr.set_title('Filtered Groundtruth')
                ax_bl.set_title('Thresholded Detections')
                ax_br.set_title('All Tracks')

                for det in curr_dets:
                    ax_bl.add_patch(patches.Rectangle((det[2], det[3]), det[4] - det[2], det[5] - det[3],
                                                      fill=False, linewidth=det[-1] + 1.5, edgecolor="red"))

                for tid, box in zip(curr_gts['TIDs'], curr_gts['boxes']):
                    vis_box = lib.box_rel2abs(box)
                    ax_tl.add_patch(patches.Rectangle((vis_box[0], vis_box[1]), vis_box[2], vis_box[3],
                                                      fill=False, linewidth=2.0, edgecolor="blue"))
            # ===/visualization===

            # ---PREDICT---
            for track in track_list:
                track.track_predict()

            num_curr_dets = len(curr_dets)
            if num_curr_dets > 0 and len(track_list) > 0:
                if args.use_appearance:
                    track_embs = np.array([track.embedding for track in track_list])
                    det_xys = [lib.box_center_xy(lib.ltrb_to_box(det[2:])) for det in curr_dets]
                    image = load_or_reuse(image, args, icam, curr_frame)
                    det_embs = embed_crops_at(net, image, det_xys,
                                              debug_out_dir=debug_dir, debug_cam=icam, debug_curr_frame=curr_frame)
                    dist_matrix = net.embeddings_cdist(track_embs, det_embs)
                    #print()
                    #print("dists-pct: {} | {} | {}".format(*np.percentile(dist_matrix.flatten(), [0, 50, 100])))
                    #print("dists-top: " + " | ".join(map(str, np.sort(dist_matrix, axis=None)[:5])))

                    # apply dist threshold here to keep munkres from finding strange compromises
                    dist_matrix = dist_matrix / APP_THRESH
                    dist_matrix[dist_matrix > 1.0] = 999999

                    # * Euclidean dist!
                    #dist_matrix_euc = np.zeros((len(track_list), num_curr_dets))
                    #for itrack, track in enumerate(track_list):
                    #    dist_matrix_euc[itrack] = [euclidean(track.KF.x[::2], lib.box_center_xy(lib.ltrb_to_box(det[2:]))) for det in curr_dets]
                    #dist_matrix_euc = dist_matrix_euc/DIST_THRESH
                    #dist_matrix_euc[dist_matrix_euc > 1.0] = 999999

                    dist_matrix = dist_matrix#*dist_matrix_euc

                else:
                    dist_matrix = np.zeros((len(track_list), num_curr_dets))

                    for itrack, track in enumerate(track_list):
                        # ---BUILD DISTANCE MATRIX---
                        #  TODO: IoU (outsource distance measure)
                        #              #dist_matrix = [euclidean(tracker.x[0::2],curr_dets[i][2:4]) for i in range(len(curr_dets))]
                        #inv_P = inv(each_tracker.KF.P[::2,::2])
                        dist_matrix[itrack] = [euclidean(track.KF.x[::2], lib.box_center_xy(lib.ltrb_to_box(det[2:]))) for det in curr_dets]
                        #              #dist_matrix_line = np.array([mahalanobis(each_tracker.KF.x[::2],
                        #                                (curr_dets[i][2]+curr_dets[i][4]/2.,
                        #                                 curr_dets[i][3]+curr_dets[i][5]/2.),
                        #                                inv_P) for i in range(len(curr_dets))])
                        #  apply the threshold here (munkres apparently can't deal 100% with inf, so use 999999)
                        #              dist_matrix_line[np.where(dist_matrix_line>dist_thresh)] = 999999
                        #              dist_matrix.append(dist_matrix_line.tolist())

                    # apply dist threshold here to keep munkres from finding strange compromises
                    dist_matrix = dist_matrix / DIST_THRESH
                    dist_matrix[dist_matrix > 1.0] = 999999

                # Do the Munkres! (Hungarian algo) to find best matching tracks<->dets
                # at first, all detections (if any) are unassigend
                unassigned_dets = set(range(num_curr_dets))

                nn_indexes = m.compute(dist_matrix.tolist())
                # perform update step for each match (check for threshold, to see, if it's actually a miss)
                for nn_match_idx in range(len(nn_indexes)):
                    # ---UPDATE---
                    if (dist_matrix[nn_indexes[nn_match_idx][0]][nn_indexes[nn_match_idx][1]] <= 1.0):
                        nn_det = curr_dets[nn_indexes[nn_match_idx][1]]  # 1st: track_idx, 2nd: 0=track_idx, 1 det_idx
                        track_list[nn_indexes[nn_match_idx][0]].track_update(lib.box_center_xy(lib.ltrb_to_box(nn_det[2:])))
                        track_list[nn_indexes[nn_match_idx][0]].track_is_matched(curr_frame)
                        # remove detection from being unassigend
                        unassigned_dets.remove(nn_indexes[nn_match_idx][1])
                    else:
                        track_list[nn_indexes[nn_match_idx][0]].track_is_missed(curr_frame)

                # set tracks without any match to miss
                for miss_idx in list(set(range(len(track_list))) - set([i[0] for i in nn_indexes])):
                    track_list[miss_idx].track_is_missed(curr_frame)

            else:  # No dets => all missed
                for track in track_list:
                    track.track_is_missed(curr_frame)


            if not args.gt_init:
                ### B) 1: get new tracks from unassigned detections
                for unassigend_det_idx in unassigned_dets:
                    if curr_dets[unassigend_det_idx][-1] > DET_INIT_THRESH:
                        init_pose = lib.box_center_xy(lib.ltrb_to_box(curr_dets[unassigend_det_idx][2:]))
                        image = load_or_reuse(image, args, icam, curr_frame)
                        new_track = Track(SEQ_DT, curr_frame, init_pose, track_id=track_id,
                                          embedding=embed_crops_at(net, image, [init_pose])[0] if args.use_appearance else None)
                        track_id = track_id + 1
                        track_list.append(new_track)
            else:
                ### B) 2: new tracks from (unassigend) ground truth
                for tid, box in zip(curr_gts['TIDs'],curr_gts['boxes']):
                    if tid in already_tracked:
                        continue
                    abs_box = lib.box_rel2abs(box)
                    init_pose = lib.box_center_xy(abs_box)
                    image = load_or_reuse(image, args, icam, curr_frame)
                    new_track = Track(SEQ_DT, curr_frame, init_pose, track_id=tid,
                                      embedding=embed_crops_at(net, image, [init_pose])[0] if args.use_appearance else None,
                                      init_thresh=1,delete_thresh=90)
                    track_list.append(new_track)
                    already_tracked.append(tid)

                    if shall_vis(args, curr_frame):
                        ax_tr.add_patch(patches.Rectangle((abs_box[0], abs_box[1]), abs_box[2], abs_box[3],
                                                          fill=False, linewidth=2.0, edgecolor="lime"))

            ### C) further track-management
            # delete tracks marked as 'deleted' in this tracking cycle
            # Modifies track_list in-place, like de-referencing a pointer in C
            track_list[:] = [i for i in track_list if i.status != 'deleted']

            # ===visualization===
            ### Plot the current state of tracks.
            if shall_vis(args, curr_frame):
                for tracker in track_list:
                    tracker.plot_track(ax_br, plot_past_trajectory=True)
                    # plt.gca().add_patch(patches.Rectangle((tracker.KF.x[0]-50, tracker.KF.x[2]-200), 100, 200,
                    #                                       fill=False, linewidth=3, edgecolor=tracker.color))

                for ax in axes:
                    ax.set_adjustable('box-forced')
                    ax.set_xlim(0, 1920)
                    ax.set_ylim(1080, 0)

                # plt.imshow(curr_heatmap,alpha=0.5,interpolation='none',cmap='hot',extent=[0,curr_image.shape[1],curr_image.shape[0],0],clim=(0, 10))
                # savefig(pjoin(args.outdir, 'camera{}/res_img_{:06d}.jpg'.format(icam, curr_frame)), quality=80)
                fig.savefig(pjoin(args.outdir, 'camera{}/res_img_{:06d}.jpg'.format(icam, curr_frame)),
                            quality=80, bbox_inches='tight', pad_inches=0.2)
                # plt.show()
                # fig.close()
                plt.close()


        # ==evaluation===
        if True:
            with open(eval_path, 'a') as eval_file:
                for icam, track_list in zip(CAMS, track_lists):
                    for tracker in track_list:
                        track_eval_line = tracker.get_track_eval_line(cid=icam,frame=curr_frame)
                        if track_eval_line is not None:
                            eval_file.write('{} {} {} {} {} {} {} {} {}\n'.format(*track_eval_line))

        global g_frames
        g_frames += 1


# Heavily adapted and fixed from http://robotics.usc.edu/~ampereir/wordpress/?p=626
def savefig(fname, fig=None, orig_size=None, **kw):
    if fig is None:
        fig = plt.gcf()
    fig.patch.set_alpha(0)

    w, h = fig.get_size_inches()
    if orig_size is not None:  # Aspect ratio scaling if required
        fw, fh = w, h
        w, h = orig_size
        fig.set_size_inches((fw, (fw/w)*h))
        fig.set_dpi((fw/w)*fig.get_dpi())

    ax = fig.gca()
    ax.set_frame_on(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_axis_off()
    #ax.set_xlim(0, w); ax.set_ylim(h, 0)
    fig.savefig(fname, transparent=True, bbox_inches='tight', pad_inches=0, **kw)

def read_detections(cams):
    print("Reading detections...")
    det_list = [[] for _ in range(len(cams))]
    for icam in cams:
        print("Camera {}...".format(icam))
        fname = pjoin(args.basedir, 'detections/camera{}_trainval-mini.mat'.format(icam))
        try:
            det_list[cams.index(icam)] = loadmat(fname)['detections']
        except NotImplementedError:
            with h5py.File(fname, 'r') as det_file:
                det_list[cams.index(icam)] = np.array(det_file['detections']).T
        # ===setup list of all detections (dukeMTMC format)===
        #with h5py.File(fname, 'r') as det_file:
        #    det_list[CAMS.index(icam)] = np.array(det_file['detections']).T
        print("done!")
    return det_list


def slice_all(f, s):
    return {k: v[s] for k,v in f.items()}

def load_trainval(fname, time_range=[49700, 227540]):
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
    start_times = [5543, 3607, 27244, 31182, 1, 22402, 18968, 46766]
    data['GFIDs'] = np.array(data['LFIDs'])
    for icam, t0 in zip(range(1,9), start_times):
        data['GFIDs'][data['Cams'] == icam] += t0 - 1

    #return data
    return slice_all(data, (time_range[0] <= data['GFIDs']) & (data['GFIDs'] <= time_range[1]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='2D tracker test.')
    parser.add_argument('--basedir', nargs='?', default='/work/breuers/dukeMTMC/',
                        help='Path to `train` folder of 2DMOT2015.')
    parser.add_argument('--outdir', nargs='?', default='/work/breuers/dukeMTMC/results/',
                        help='Where to store generated output. Only needed if `--vis` is also passed.')
    parser.add_argument('--use_appearance', action='store_true',
                        help='Whether or not to use the deep net as appearance model.')
    parser.add_argument('--model', default='lunet2c',
                        help='Name of the model to load. Corresponds to module names in lib/models. Or `fake`')
    parser.add_argument('--weights', default='/work/breuers/dukeMTMC/models/lunet2c-noscale-nobg-2to32-aug.pkl',
                        help='Name of the weights to load for the model (path to .pkl file).')
    parser.add_argument('--t0', default=127720, type=int,
                        help='Time of first frame.')
    parser.add_argument('--t1', default=187540, type=int,
                        help='Time of last frame, inclusive.')
    parser.add_argument('--vis', default=0, type=int,
                        help='Generate and save visualization of the results, every X frame.')
    parser.add_argument('--debug', action='store_true',
                        help='Generate extra many debugging outputs (in outdir).')
    parser.add_argument('--gt_init', action='store_true',
                        help='Use first groundtruth to init tracks.')
    parser.add_argument('--cams', default='1,2,3,4,5,6,7,8',
                        help='Array of cameras numbers (1-8) to consider.')
    args = parser.parse_args()
    args.cams = eval('[' + args.cams + ']')
    print(args)

    # This is all for faking the network.
    net = SemiFakeNews(model=args.model, weights=args.weights,
                       input_scale_factor=1.0 if SCALE_FACTOR==0.5 else 0.5,  # ASK LUCAS
                       debug_skip_full_image=True,  # Goes with the above.
                       fake_dets=None,
                       fake_shape=None,
                       ) if args.use_appearance else None

    # Prepare output dirs
    for icam in args.cams:
        makedirs(pjoin(args.outdir, 'camera{}'.format(icam)), exist_ok=True)
    makedirs(pjoin(args.outdir, 'results'), exist_ok=True)

    tstart = time.time()
    try:
        main(net, args)
    except KeyboardInterrupt:
        print()

    print('FPS: {:.3f}'.format(g_frames / (time.time() - tstart)))
