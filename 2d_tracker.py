#!/usr/bin/env python

# the usual suspects
import numpy as np
import numpy.random as rnd
import matplotlib
matplotlib.use('GTK')
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.patches as patches

#tracker stuff
from filterpy.kalman import KalmanFilter
from track import Track
from scipy.linalg import block_diag,inv
from filterpy.common import Q_discrete_white_noise
from filterpy.stats import plot_covariance_ellipse

#other stuff
from scipy.spatial.distance import euclidean,mahalanobis
from munkres import Munkres, print_matrix
import time

# ===init sequence===
work_dir = '/home/stefan/projects/MOTChallenge/2DMOT2015/train/'
#TODO: have a sequence object instead of the below?
all_sequences = ['ADL-Rundle-6', 'ADL-Rundle-8', 'ETH-Bahnhof', 'ETH-Pedcross2', 'ETH-Sunnyday', 'KITTI-13',
             'KITTI-17', 'PETS09-S2L1', 'TUD-Campus', 'TUD-Stadtmitte', 'Venice-2']
all_frames = [525,654,1000,837,354,340,145,795,71,179,600]
all_fps = [30,30,14,14,14,10,10,7,25,25,30]
all_resolutions = [[1920,1080],[1920,1080],[640,480],[640,480],[640,480],[1224,370],
                   [1224,370],[768,576],[640,480],[640,480],[1920,1080]]
# CHOOSE SEQUENCE HERE
seq_num = 0 # ADL-Rundle-8 ETH-Bahnhof ETH-Pedcross2 ETH-Sunnyday KITTI-13 KITTI-17 PETS09-S2L1 TUD-Campus TUD-Stadtmitte Venice-2
# set sequence info
seq_dir = work_dir + all_sequences[seq_num]
seq_fps = all_fps[seq_num]
seq_frames = all_frames[seq_num]
seq_resolution = all_resolutions[seq_num]

# ===setup list of all detections (MOT format)===
det_string = seq_dir+'/det/det.txt'
with open(det_string, 'r') as det_file:
    # create and fill list of all detections #TODO: special det object, to handle status, modality,...
    det_list = []
    for det_line in det_file:
        det_line = det_line.rstrip('\n')
        one_det = det_line.split(',')
        # check valid line
        if (len(one_det) == 10):
            one_det[0] = int(one_det[0]) #TODO: nicer way to format this?
            one_det[1] = int(one_det[1])
            one_det[2] = int(one_det[2])
            one_det[3] = int(one_det[3])
            one_det[4] = float(one_det[4])
            one_det[5] = float(one_det[5])
            one_det[6] = float(one_det[6])
            one_det[7] = float(one_det[7])
            one_det[8] = float(one_det[8])
            one_det[9] = float(one_det[9])
            det_list.append(one_det)
        else:
            print 'Warning: misformed detection line according to MOT format (10 entries needed)'
    # > 'det_list[x][y]',  to access detection x, data y
    # one detection line in 2D-MOTChallenge format:
    # field     0        1     2          3         4           5            6       7    8    9
    # data      <frame>  <id>  <bb_left>  <bb_top>  <bb_width>  <bb_height>  <conf>  <x>  <y>  <z>
first_dets = [x for x in det_list if x[0]==1]


# ===init tracks and other stuff==
do_vis = True
track_id = 1
dt = 1./seq_fps
track_list = []
# debug: init one tracker for each first detection #TODO: only start tracks, when min_num_dets (also down below)
for first_det_idx in xrange(len(first_dets)):
    init_x = [first_dets[first_det_idx][2]+first_dets[first_det_idx][4]/2., 0,
                 first_dets[first_det_idx][3]+first_dets[first_det_idx][5]/2., 0]
    init_P = np.eye(4)*1000
    new_track = Track(init_x,init_P,dt,1,track_id=track_id)
    track_id = track_id + 1
    track_list.append(new_track)
# init munkres (=Hungarian Algorithm) to find NN in DA step #TODO: IoU (outsource distance measure)
m = Munkres()
dist_thresh = 20 #pixel #TODO: dependent on resolution

tstart = time.time()
# ===Tracking fun begins: iterate over frames===
for curr_frame in xrange(1,seq_frames+1):
    # get detections in current frame
    curr_dets = [x for x in det_list if x[0]==curr_frame]
    num_curr_dets = len(curr_dets)
    # > 'curr_dets[0][2:4]' #to get only bb_left and bb_top
    # > 'curr_dets[0][2]+curr_dets[0][4]/2.,curr_dets[0][3]+curr_dets[0][5]/2.' #to get center point of det_box 0

    # init/reset distance matrix for later
    dist_matrix = [] #np.array([])
    # loop over trackers (predict -> distance_matrix -> update)
    for each_tracker in track_list:
        # ---PREDICT---
        each_tracker.KF.predict()

        # no detections? no distance matrix
        if not num_curr_dets:
            print 'No detections in this frame.'
            break
        # ---BUILD DISTANCE MATRIX---
        #dist_matrix = [euclidean(tracker.x[0::2],curr_dets[i][2:4]) for i in range(len(curr_dets))]
        dist_matrix_line = np.array([mahalanobis(each_tracker.KF.x[::2],
                                        (curr_dets[i][2]+curr_dets[i][4]/2.,
                                         curr_dets[i][3]+curr_dets[i][5]/2.),
                                        inv(each_tracker.KF.P[::2,::2])) for i in range(len(curr_dets))])
        # apply the threshold here (munkres apparently can't deal 100% with inf, so use 999999)
        dist_matrix_line[np.where(dist_matrix_line>dist_thresh)] = 999999
        dist_matrix.append(dist_matrix_line.tolist())

    # Do the Munkres! (Hungarian algo) to find best matching tracks<->dets
    # at first, all detections (if any) are unassigend
    unassigned_dets = set(range(num_curr_dets))
    if len(dist_matrix) != 0:
        nn_indexes = m.compute(dist_matrix)
        # perform update step for each match (check for threshold, to see, if it's actually a miss)
        for nn_match_idx in xrange(len(nn_indexes)):
            # ---UPDATE---
            if (dist_matrix[nn_indexes[nn_match_idx][0]][nn_indexes[nn_match_idx][1]]<=dist_thresh):
                nn_det = curr_dets[nn_indexes[nn_match_idx][1]] #1st: track_idx, 2nd: 0=track_idx, 1 det_idx
                track_list[nn_indexes[nn_match_idx][0]].KF.update([nn_det[2] + nn_det[4] / 2., nn_det[3] + nn_det[5]/2.])
                track_list[nn_indexes[nn_match_idx][0]].track_is_matched(curr_frame)
                # remove detection from being unassigend
                unassigned_dets.remove(nn_indexes[nn_match_idx][1])
            else:
                track_list[nn_indexes[nn_match_idx][0]].track_is_missed(curr_frame)
        # set tracks without any match to miss
        for miss_idx in list(set(range(len(track_list))) - set([i[0] for i in nn_indexes])):
            track_list[miss_idx].track_is_missed(curr_frame)

    # ===track management===
    # start new tracks for unassigend dets (really unassigend + not assigend due to thresh, set above)
    # TODO: min_det_num dets needed to really 'start' new tracks (see above) status 'active/inactive'
    for unassigend_det_idx in unassigned_dets:
        init_x = [curr_dets[unassigend_det_idx][2] + curr_dets[unassigend_det_idx][4]/2., 0,
                  curr_dets[unassigend_det_idx][3] + curr_dets[unassigend_det_idx][5]/2., 0]
        init_P = np.eye(4) * 1000
        new_track = Track(init_x, init_P, dt, curr_frame, track_id=track_id)
        track_id = track_id + 1
        track_list.append(new_track)
    # delete tracks marked as 'deleted' in this tracking cycle #TODO: manage in other list for re-id
    track_list = [i for i in track_list if i.status != 'deleted']
    # ... sth. else?

    # ===visualization===
    if do_vis:
        # open image file
        image_string = seq_dir+'/img1/{:06d}.jpg'.format(curr_frame) #TODO: are all images .jpgs? ifn put as seq_attr
        image_file = cbook.get_sample_data(image_string)
        image_save_path = '/home/stefan/results/2d_tracker/res_img_{:06d}'.format(curr_frame)
        curr_image = plt.imread(image_file)
        # > 'image[50:250,50:250] = 255' #simple image manipulations
        # plot detections
        for det in curr_dets:
            plt.gca().add_patch(patches.Rectangle((det[2],det[3]),det[4],det[5],fill=False,linewidth=det[6]/10.0,edgecolor="red"))
            #plot_covariance_ellipse((det[2]+det[4]/2.,det[3]+det[5]/2.),[[200,0],[0,200]],fc='r',alpha=0.4,std=[1,2,3])
        # plot (active) tracks
        for each_tracker in track_list:
            each_tracker.plot_track(plot_past_trajectory=True)
            #plt.gca().add_patch(patches.Rectangle((each_tracker.KF.x[0]-50, each_tracker.KF.x[2]-200),
            #                                        100, 200, fill=False, linewidth=5, edgecolor=each_tracker.color))
        plt.imshow(curr_image)
        plt.savefig(image_save_path)
        #plt.show()
        plt.close()

print 'FPS:', seq_frames / (time.time() - tstart)

