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

#tracker stuff
import lib
from track import Track

#other stuff
from scipy.spatial.distance import euclidean,mahalanobis
import json
from pprint import pprint

SEQ_FPS = 60.0
SEQ_SHAPE = (1080, 1920)
STATE_SHAPE = (270, 480)
HEATMAP_SHAPE = (36, 64)


# ===functions===
def heatmap_sampling_for_dets(heatmap, dets_boxes):
    H, W = heatmap.shape
    for l, t, w, h in dets_boxes:
        # score is how many times more samples than pixels in the detection box.
        score = random.randint(1,5)
        add_idx = np.random.multivariate_normal([l+w/2, t+h/2], [[(w/6)**2, 0], [0, (h/6)**2]], int(np.prod(heatmap.shape)*h*w*score))
        np.add.at(heatmap, [[int(np.clip(y, 0, 0.999)*H) for x,y in add_idx],
                            [int(np.clip(x, 0, 0.999)*W) for x,y in add_idx]], 1)
    return heatmap


parser = argparse.ArgumentParser(description='2D tracker test.')
parser.add_argument('--basedir', nargs='?', default='/work/breuers/dukeMTMC/',
                    help='Path to `train` folder of 2DMOT2015.')
parser.add_argument('--outdir', nargs='?', default='/home/breuers/results/duke_mtmc/',
                    help='Where to store generated output. Only needed if `--vis` is also passed.')
parser.add_argument('--vis', action='store_true', default=True,
                    help='Generate and save visualization of the results.')
args = parser.parse_args()
print(args)

dets = lib.load_trainval(pjoin(args.basedir, 'ground_truth', 'trainval.mat'))

# ===init tracks and other stuff==
track_id = 1
dt = 1./SEQ_FPS

track_lists = [[], [], [], [], [], [], [], []]

dist_thresh = 100 #pixel #TODO: dependent on resolution

# Prepare output dirs
for icam in range(1, 8+1):
    makedirs(pjoin(args.outdir, 'camera{}'.format(icam)), exist_ok=True)

class FakeNews:
    def __init__(self):
        self.already_tracked_ids = [[], [], [], [], [], [], [], []]

    def embed_image(self, image):
        return None  # z.B. (30,60,128)


    def search_person(self, image_embedding, person_embedding, fake_track_id, fake_curr_cam_dets):
        id_heatmap = np.random.rand(*HEATMAP_SHAPE)
        id_det_boxes = fake_curr_cam_dets['boxes'][fake_curr_cam_dets['TIDs'] == fake_track_id]
        return heatmap_sampling_for_dets(id_heatmap, id_det_boxes)


    def personness(self, image, known_embeddings,
                   fake_curr_dets, fake_icam):
        curr_cam_dets = lib.slice_all(fake_curr_dets, fake_curr_dets['Cams'] == fake_icam)
        new_det_indices = np.where(np.logical_not(np.in1d(curr_cam_dets['TIDs'], self.already_tracked_ids[icam - 1])))[0]
        new_heatmaps_and_ids = []
        for each_det_idx in new_det_indices:
            new_heatmap = np.random.rand(*HEATMAP_SHAPE)
            new_heatmap = heatmap_sampling_for_dets(new_heatmap, [curr_cam_dets['boxes'][each_det_idx]])
            new_id = curr_cam_dets['TIDs'][each_det_idx]
            # TODO: get correct track_id (loop heatmap, instead of function call?# )
            # TODO: get id_heatmap of that guy for init_heatmap
            self.already_tracked_ids[icam - 1].append(new_id)
            new_heatmaps_and_ids.append((new_heatmap, new_id))
        return new_heatmaps_and_ids


#@profile
def main():
    tstart = time.time()

    fake_neural_news_network = FakeNews()

    # ===Tracking fun begins: iterate over frames===
    # TODO: global time (duke)
    for curr_frame in range(49700, 227540+1):
        print("\rFrame {}, {} tracks".format(curr_frame, list(map(len, track_lists))), end='', flush=True)

        images = [plt.imread(pjoin(args.basedir, 'frames/camera{}/{}.jpg'.format(icam, lib.glob2loc(curr_frame, icam)))) for icam in range(1,8+1)]

        curr_dets = lib.slice_all(dets, dets['GFIDs'] == curr_frame)
        num_curr_dets = len(curr_dets)

        image_embeddings = list(map(fake_neural_news_network.embed_image, images))

        for icam, track_list in zip(range(1, 8+1), track_lists):
            curr_cam_dets = lib.slice_all(curr_dets, curr_dets['Cams'] == icam)
            ### A) update existing tracks
            for itracker, each_tracker in enumerate(track_list):
                # get ID_heatmap
                id_heatmap = fake_neural_news_network.search_person(image_embeddings[icam-1], each_tracker.embedding,
                                                                    fake_track_id=each_tracker.track_id, fake_curr_cam_dets=curr_cam_dets)
                # ---PREDICT---
                each_tracker.track_predict()
                # ---UPDATE---
                each_tracker.track_update(id_heatmap, curr_frame, images[icam-1])



        ### B) get new tracks from general heatmap
        for icam in range(1, 8 + 1):
            # TODO: ID management (duke)
            for new_heatmap, new_id in fake_neural_news_network.personness(images[icam-1], known_embeddings=None,
                                                                           fake_curr_dets=curr_dets, fake_icam=icam):
                # TODO: get correct track_id (loop heatmap, instead of function call?# )
                # TODO: get id_heatmap of that guy for init_heatmap
                new_track = Track(dt, curr_frame, new_heatmap, images[icam-1], track_id=new_id,
                                  state_shape=STATE_SHAPE, output_shape=SEQ_SHAPE)
                track_lists[icam-1].append(new_track)


        ### C) further track-management
        # delete tracks marked as 'deleted' in this tracking cycle #TODO: manage in other list for re-id
        #track_list = [i for i in track_list if i.status != 'deleted']

        # ===visualization===
        if args.vis:
            for icam, track_list in zip(range(1, 8 + 1), track_lists):
                # open image file
                #curr_image = plt.imread(pjoin(args.basedir, 'frames/camera{}/{}.jpg'.format(icam, lib.glob2loc(curr_frame, icam))))  # TODO
                curr_image = images[icam-1]
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
