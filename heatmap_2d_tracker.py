#!/usr/bin/env python
# -*- coding: utf-8 -*-.

from __future__ import print_function
from __future__ import division

import argparse
from os.path import join as pjoin
from os import makedirs
import time

# the usual suspects
import numpy as np
import random
import scipy
import numpy.random as rnd
import matplotlib as mpl
#mpl.use('Agg')
#mpl.use('GTK')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import h5py
from scipy.io import loadmat


#tracker stuff
from track import Track

#other stuff
from scipy.spatial.distance import euclidean,mahalanobis
import json
from pprint import pprint

START_TIMES = [5543, 3607, 27244, 31182, 1, 22402, 18968, 46766]
TRAIN_START, TRAIN_END = 49700, 227540
SEQ_FPS = 60.0
SEQ_SHAPE = (1080, 1920)
STATE_SHAPE = (270, 480)
HEATMAP_SHAPE = (36, 64)


# ===functions===
def loc2glob(loc, cam):
    # Compute global frame numbers once.
    offset = START_TIMES[cam-1] - 1
    return loc + offset


def glob2loc(glob, cam):
    # Compute global frame numbers once.
    offset = START_TIMES[cam-1] - 1
    return glob - offset


def heatmap_sampling_for_dets(heatmap, dets_boxes):
    H, W = heatmap.shape
    for l, t, w, h in dets_boxes:
        # score is how many times more samples than pixels in the detection box.
        score = random.randint(1,5)
        add_idx = np.random.multivariate_normal([l+w/2, t+h/2], [[(w/6)**2, 0], [0, (h/6)**2]], int(np.prod(heatmap.shape)*h*w*score))
        np.add.at(heatmap, [[int(np.clip(y, 0, 0.999)*H) for x,y in add_idx],
                            [int(np.clip(x, 0, 0.999)*W) for x,y in add_idx]], 1)
    return heatmap


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
    data['boxes'][:,0] /= SEQ_SHAPE[1]
    data['boxes'][:,1] /= SEQ_SHAPE[0]
    data['boxes'][:,2] /= SEQ_SHAPE[1]
    data['boxes'][:,3] /= SEQ_SHAPE[0]

    # Compute global frame numbers once.
    data['GFIDs'] = np.array(data['LFIDs'])
    for icam, t0 in zip(range(1,9), START_TIMES):
        data['GFIDs'][data['Cams'] == icam] += t0 - 1

    return slice_all(data, (time_range[0] <= data['GFIDs']) & (data['GFIDs'] <= time_range[1]))

parser = argparse.ArgumentParser(description='2D tracker test.')
parser.add_argument('--basedir', nargs='?', default='/work/breuers/dukeMTMC/',
                    help='Path to `train` folder of 2DMOT2015.')
parser.add_argument('--outdir', nargs='?', default='/home/breuers/results/duke_mtmc/',
                    help='Where to store generated output. Only needed if `--vis` is also passed.')
parser.add_argument('--vis', action='store_true', default=True,
                    help='Generate and save visualization of the results.')
args = parser.parse_args()
print(args)

dets = load_trainval(pjoin(args.basedir, 'ground_truth', 'trainval.mat'), time_range=(TRAIN_START, TRAIN_END))

    # > 'det_list[x][y]',  to access detection x, data y
    # one detection line in 2D-MOTChallenge format:
    # field     0        1     2          3         4           5            6       7    8    9
    # data      <frame>  <id>  <bb_left>  <bb_top>  <bb_width>  <bb_height>  <conf>  <x>  <y>  <z>

# ===init tracks and other stuff==
track_id = 1
dt = 1./SEQ_FPS

track_lists = [[], [], [], [], [], [], [], []]
already_tracked_ids = [[], [], [], [], [], [], [], []]

dist_thresh = 100 #pixel #TODO: dependent on resolution

# Prepare output dirs
for icam in range(1, 8+1):
    makedirs(pjoin(args.outdir, 'camera{}'.format(icam)), exist_ok=True)

#@profile
def main():
    tstart = time.time()
    # ===Tracking fun begins: iterate over frames===
    # TODO: global time (duke)
    for curr_frame in range(49700, 227540+1):
        print("\rFrame {}, {} tracks".format(curr_frame, list(map(len, track_lists))), end='', flush=True)

        curr_dets = slice_all(dets, dets['GFIDs'] == curr_frame)
        num_curr_dets = len(curr_dets)

        for icam, track_list in zip(range(1, 8+1), track_lists):
            curr_cam_dets = slice_all(curr_dets, curr_dets['Cams'] == icam)
            ### A) update existing tracks
            for itracker, each_tracker in enumerate(track_list):
                # get ID_heatmap
                id_heatmap = np.random.rand(*HEATMAP_SHAPE)
                id_det_boxes = curr_cam_dets['boxes'][curr_cam_dets['TIDs'] == each_tracker.track_id]
                id_heatmap = heatmap_sampling_for_dets(id_heatmap, id_det_boxes)
                # ---PREDICT---
                each_tracker.track_predict()
                # ---UPDATE---
                each_tracker.track_update(id_heatmap)
                if each_tracker.pos_heatmap.max() > 2.0:
                    each_tracker.track_is_matched(curr_frame)
                else:
                    each_tracker.track_is_missed(curr_frame)



        ### B) get new tracks from general heatmap
        for icam in range(1, 8 + 1):
            # TODO: ID management (duke)
            curr_cam_dets = slice_all(curr_dets, curr_dets['Cams'] == icam)
            new_det_indices = np.where(np.logical_not(np.in1d(curr_cam_dets['TIDs'], already_tracked_ids[icam-1])))[0]
            for each_det_idx in new_det_indices:
                new_heatmap = np.random.rand(*HEATMAP_SHAPE)
                #if icam == 2:
                #    import ipdb ; ipdb.set_trace()
                new_heatmap = heatmap_sampling_for_dets(new_heatmap, [curr_cam_dets['boxes'][each_det_idx]])
                new_id = curr_cam_dets['TIDs'][each_det_idx]
                # TODO: get correct track_id (loop heatmap, instead of function call?# )
                # TODO: get id_heatmap of that guy for init_heatmap
                new_track = Track(dt, curr_frame, new_heatmap, track_id=new_id,
                                  state_shape=STATE_SHAPE, output_shape=SEQ_SHAPE)
                track_lists[icam-1].append(new_track)
                already_tracked_ids[icam-1].append(new_id)


        ### C) further track-management
        # delete tracks marked as 'deleted' in this tracking cycle #TODO: manage in other list for re-id
        #track_list = [i for i in track_list if i.status != 'deleted']

        # ===visualization===
        if args.vis:
            for icam, track_list in zip(range(1, 8 + 1), track_lists):
                # open image file
                curr_image = plt.imread(pjoin(args.basedir, 'frames/camera{}/{}.jpg'.format(icam, glob2loc(curr_frame, icam))))  # TODO
                plt.imshow(curr_image)

                # plot (active) tracks
                for each_tracker in track_list:
                    #if(each_tracker.track_id==3):
                    each_tracker.plot_track(plot_past_trajectory=True, plot_heatmap=True)
                    #break
                    #plt.gca().add_patch(patches.Rectangle((each_tracker.KF.x[0]-50, each_tracker.KF.x[2]-200),
                    #                                        100, 200, fill=False, linewidth=3, edgecolor=each_tracker.color))

                #plt.imshow(curr_heatmap,alpha=0.5,interpolation='none',cmap='hot',extent=[0,curr_image.shape[1],curr_image.shape[0],0],clim=(0, 10))
                plt.savefig(pjoin(args.outdir, 'camera{}/res_img_{:06d}.jpg'.format(icam, curr_frame)))
                #plt.show()
                plt.close()

    print('FPS: {:.3f}'.format((TRAIN_END+1 - TRAIN_START) / (time.time() - tstart)))


if __name__ == '__main__':
    main()
