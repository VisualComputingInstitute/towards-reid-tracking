#!/usr/bin/env python
# -*- coding: utf-8 -*-.

from __future__ import print_function
from __future__ import division

import argparse
from os.path import join as pjoin
import time

# the usual suspects
import numpy as np
import random
import scipy
import numpy.random as rnd
import matplotlib
#matplotlib.use('GTK')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.mlab as mlab
import h5py
from scipy.io import loadmat


#tracker stuff
from filterpy.kalman import KalmanFilter
from track import Track
from scipy.linalg import block_diag,inv
from filterpy.common import Q_discrete_white_noise
from filterpy.stats import plot_covariance_ellipse

#other stuff
from scipy.spatial.distance import euclidean,mahalanobis
from munkres import Munkres, print_matrix
import json
from pprint import pprint

# ===functions===
def heatmap_sampling_for_dets(heatmap, heatmap_scale, seq_shape, dets):
    for each_det in dets:
        cut_bb_top = max(0, round(each_det[3]))/heatmap_scale
        cut_bb_left = max(0, round(each_det[2]))/heatmap_scale
        cut_bb_bottom = min(seq_shape[0] - 1, round(each_det[3] + each_det[5]-1))/heatmap_scale
        cut_bb_right = min(seq_shape[1] - 1, round(each_det[2] + each_det[4]-1))/heatmap_scale
        cut_h = (cut_bb_bottom - cut_bb_top)
        cut_w = (cut_bb_right - cut_bb_left)
        center_x = round(cut_bb_top+cut_h/2)
        center_y =  round(cut_bb_left+cut_w/2)
        score = random.randint(10,50)# max(0,each_det[6])
        #x_range = np.arange(cut_w)
        #y_range = np.arange(cut_h)
        #X_grid,Y_grid = np.meshgrid(x_range,y_range)
        #Z = mlab.bivariate_normal(X_grid,Y_grid,15,45,X_grid.shape[1]/2,X_grid.shape[0]/2)
        #curr_heatmap[min(seq.shape[1]-1,round(each_det[3]+each_det[5]/2.)),min(seq.shape[0]-1,round(each_det[2]+each_det[4]/2.))] = 255
        #curr_heatmap[cut_bb_top:cut_bb_bottom, cut_bb_left:cut_bb_right] += Z*score
        add_idx = np.random.multivariate_normal([center_x, center_y], [[cut_h/heatmap_scale, 0], [0, cut_w/heatmap_scale]], int(score/heatmap_scale*2*cut_h))
        np.add.at(heatmap,[[max(0,int(add_idx[i][0])) for i in range(len(add_idx))], [max(0,int(add_idx[i][1])) for i in range(len(add_idx))]],1)
    return heatmap

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

# ===init sequence===
class Sequence(object):
    def __init__(self, name, nframes, fps, width, height):
        self.name = name
        self.nframes = nframes
        self.fps = fps
        self.shape = (height, width)  # NOTE: (H,W) not (W,H) for consistency with numpy!

all_sequences = {
    #3d test
    'AVG-TownCentre': Sequence('AVG-TownCentre', nframes=450, fps=2.5, width=1920, height=1080),
}

parser = argparse.ArgumentParser(description='2D tracker test.')
parser.add_argument('--traindir', nargs='?', default='/home/breuers/data/MOTChallenge/3DMOT2015/test/',
                    help='Path to `train` folder of 2DMOT2015.')
parser.add_argument('--outdir', nargs='?', default='/home/breuers/results/duke_mtmc/',
                    help='Where to store generated output. Only needed if `--vis` is also passed.')
parser.add_argument('--sequence', nargs='?', choices=all_sequences.keys(), default='AVG-TownCentre')
parser.add_argument('--vis', action='store_true', default=True,
                    help='Generate and save visualization of the results.')
args = parser.parse_args()
print(args)

seq = all_sequences[args.sequence]
seq_dir = pjoin(args.traindir, seq.name)

# ===setup list of all detections (MOT format)===
with open(pjoin(seq_dir, 'gt/gt.txt'), 'r') as det_file:
    # create and fill list of all detections #TODO: special det object, to handle status, modality,...
    det_list = []
    # == MOTCHALLENGE format ==
    for det_line in det_file:
        det_line = det_line.rstrip('\n')
        one_det = det_line.split(',')
        # check valid line
        if (len(one_det) == 10):
            one_det[0] = int(one_det[0])+1 #TODO: nicer way to format this?
            one_det[1] = int(one_det[1])
            one_det[2] = float(one_det[2])
            one_det[3] = float(one_det[3])
            one_det[4] = float(one_det[4])
            one_det[5] = float(one_det[5])
            one_det[6] = float(one_det[6])
            one_det[7] = float(one_det[7])
            one_det[8] = float(one_det[8])
            one_det[9] = float(one_det[9])
            det_list.append(one_det)
        else:
            print('Warning: misformed detection line according to MOT format (10 entries needed)')

    det_list = np.array(det_list)
                    # > 'det_list[x][y]',  to access detection x, data y
    # one detection line in 2D-MOTChallenge format:
    # field     0        1     2          3         4           5            6       7    8    9
    # data      <frame>  <id>  <bb_left>  <bb_top>  <bb_width>  <bb_height>  <conf>  <x>  <y>  <z>
#first_dets = [x for x in det_list if x[0]==1]

# ===init tracks and other stuff==
track_id = 1
dt = 1./seq.fps
track_list = []
m = Munkres()
dist_thresh = 100 #pixel #TODO: dependent on resolution
heatmap_scale = 20
already_tracked_ids = []

tstart = time.time()
# ===Tracking fun begins: iterate over frames===
# TODO: global time (duke)
for curr_frame in range(1,seq.nframes+1):
    # get detections in current frame
    curr_dets = det_list[det_list[:,0] == curr_frame] # [x for x in det_list if x[0]==curr_frame]
    num_curr_dets = len(curr_dets)

    # TODO: cam loop (duke)
    ### A) update existing tracks
    for each_tracker in track_list:
        # get ID_heatmap
        id_heatmap = np.random.rand(int(seq.shape[0]/heatmap_scale),int(seq.shape[1]/heatmap_scale))
        id_det = [x for x in curr_dets if x[1] == each_tracker.track_id]
        id_heatmap = heatmap_sampling_for_dets(id_heatmap,heatmap_scale,seq.shape,id_det)
        id_heatmap = scipy.misc.imresize(id_heatmap, seq.shape, interp='bicubic', mode='F')
        # ---PREDICT---
        each_tracker.track_predict()
        # ---UPDATE---
        each_tracker.track_update(id_heatmap)
        if each_tracker.pos_heatmap.max()>2.0:
            each_tracker.track_is_matched(curr_frame)
        else:
            each_tracker.track_is_missed(curr_frame)



    ### B) get new tracks from general heatmap
    # setup heatmap
    #curr_heatmap = np.random.rand(int(seq.shape[0]/heatmap_scale),int(seq.shape[1]/heatmap_scale))
    #curr_heatmap = np.zeros(seq.shape)
    # > 'curr_dets[0][2:4]' #to get only bb_left and bb_top
    # > 'curr_dets[0][2]+curr_dets[0][4]/2.,curr_dets[0][3]+curr_dets[0][5]/2.' #to get center point of det_box 0
    # fill heatmap with det (not with already tracked ones!)
    # TODO: cam loop (duke)
    # TODO: ID management (duke)
    for each_det in curr_dets:
        if each_det[1] in already_tracked_ids:
            continue
        new_heatmap = np.random.rand(int(seq.shape[0] / heatmap_scale), int(seq.shape[1] / heatmap_scale))
        new_heatmap = heatmap_sampling_for_dets(new_heatmap, heatmap_scale, seq.shape, [each_det])
        new_heatmap = scipy.misc.imresize(new_heatmap, seq.shape, interp='bicubic', mode='F')
        new_peak = np.unravel_index(new_heatmap.argmax(), new_heatmap.shape)
        start_pose = [new_peak[1], new_peak[0]]
        # print(start_pose)
        init_x = [0.0, 0.0]
        init_P = [[10.0, 0], [0, 10.0]]
        new_id = each_det[1]
        # TODO: get correct track_id (loop heatmap, instead of function call?# )
        # TODO: get id_heatmap of that guy for init_heatmap
        new_track = Track(init_x, init_P, dt, curr_frame, start_pose, new_heatmap, track_id=new_id)
        track_list.append(new_track)
        already_tracked_ids.append(new_id)


    #new_peaks_idc = np.where(curr_heatmap>5.0)
    #if new_peaks_idc[0].any():
        #for peak_idx in range(len(new_peaks_idc[0])-1):
            # start_pose = [new_peaks_idc[1][peak_idx]*20,new_peaks_idc[0][peak_idx]*20]
            # #print(start_pose)
            # init_x = [0.0,0.0]
            # init_P = [[10.0,0],[0,10.0]]
            # #TODO: get correct track_id (loop heatmap, instead of function call?# )
            # #TODO: get id_heatmap of that guy for init_heatmap
            # new_track = Track(init_x, init_P, dt, curr_frame, start_pose, curr_heatmap, track_id=track_id)
            # track_id = track_id + 1
            # track_list.append(new_track)
            # already_tracked_ids.append(track_id)

    ### C) further track-management
    # delete tracks marked as 'deleted' in this tracking cycle #TODO: manage in other list for re-id
    track_list = [i for i in track_list if i.status != 'deleted']

    # ===visualization===
    if args.vis:
        # open image file
        curr_image = plt.imread(pjoin(seq_dir, 'img1/{:06d}.jpg'.format(curr_frame)))
        plt.imshow(curr_image)

        # > 'image[50:250,50:250] = 255' #simple image manipulations
        # plot detections
        #for det in curr_dets:
        #    plt.gca().add_patch(patches.Rectangle((det[2],det[3]),det[4],det[5],fill=False,linewidth=det[6]/10.0,edgecolor="red"))
            #plot_covariance_ellipse((det[2]+det[4]/2.,det[3]+det[5]/2.),[[200,0],[0,200]],fc='r',alpha=0.4,std=[1,2,3])

        # plot (active) tracks
        for each_tracker in track_list:
            #if(each_tracker.track_id==3):
            each_tracker.plot_track(plot_past_trajectory=True,plot_heatmap=False)
            #break
            #plt.gca().add_patch(patches.Rectangle((each_tracker.KF.x[0]-50, each_tracker.KF.x[2]-200),
            #                                        100, 200, fill=False, linewidth=3, edgecolor=each_tracker.color))

        #plt.imshow(curr_heatmap,alpha=0.5,interpolation='none',cmap='hot',extent=[0,curr_image.shape[1],curr_image.shape[0],0],clim=(0, 10))
        plt.savefig(pjoin(args.outdir, 'res_img_{:06d}'.format(curr_frame)))
        #plt.show()
        plt.close()

print('FPS: {:.3f}'.format(seq.nframes / (time.time() - tstart)))
