#!/usr/bin/env python
# -*- coding: utf-8 -*-.

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

#tracker stuff
import lib
from track import Track
from fakenews import FakeNeuralNewsNetwork


SEQ_FPS = 60.0
SEQ_DT = 1./SEQ_FPS
SEQ_SHAPE = (1080, 1920)
STATE_SHAPE = (270, 480)


g_frames = 0  # Global counter for correct FPS in all cases


#@profile
def main(net):
    track_lists = [[], [], [], [], [], [], [], []]

    # ===Tracking fun begins: iterate over frames===
    # TODO: global time (duke)
    for curr_frame in range(49700, 227540+1):
        print("\rFrame {}, {} tracks".format(curr_frame, list(map(len, track_lists))), end='', flush=True)
        net.tick(curr_frame)

        images = [plt.imread(pjoin(args.basedir, 'frames/camera{}/{}.jpg'.format(icam, lib.glob2loc(curr_frame, icam)))) for icam in range(1,8+1)]

        image_embeddings = list(map(net.embed_image, images))

        for icam, track_list in zip(range(1, 8+1), track_lists):
            net.fake_camera(icam)

            ### A) update existing tracks
            for itracker, each_tracker in enumerate(track_list):
                # get ID_heatmap
                id_heatmap = net.search_person(image_embeddings[icam-1], each_tracker.embedding,
                                               fake_track_id=each_tracker.track_id)
                # ---PREDICT---
                each_tracker.track_predict()
                # ---UPDATE---
                each_tracker.track_update(id_heatmap, curr_frame, images[icam-1])



        ### B) get new tracks from general heatmap
        for icam in range(1, 8 + 1):
            net.fake_camera(icam)

            # TODO: ID management (duke)
            for new_heatmap, new_id in net.personness(images[icam-1], known_embs=None):
                # TODO: get correct track_id (loop heatmap, instead of function call?# )
                # TODO: get id_heatmap of that guy for init_heatmap
                new_track = Track(net.embed_crop, SEQ_DT,
                                  curr_frame, new_heatmap, images[icam-1], track_id=new_id,
                                  state_shape=STATE_SHAPE, output_shape=SEQ_SHAPE)
                track_lists[icam-1].append(new_track)


        ### C) further track-management
        # delete tracks marked as 'deleted' in this tracking cycle #TODO: manage in other list for re-id
        #track_list = [i for i in track_list if i.status != 'deleted']

        # ==evaluation===
        if (True):
            with open(pjoin(args.basedir, 'results/run_{:%Y-%m-%d_%H:%M:%S}.txt'.format(datetime.datetime.now()), 'a')) as eval_file:
                for icam, track_list in zip(range(1, 8 + 1), track_lists):
                    for each_tracker in track_list:
                        track_eval_line = each_tracker.get_track_eval_line(cid=icam,frame=curr_frame)
                        eval_file.write('{} {} {} {} {} {} {} {} {}\n'.format(*track_eval_line))

        # ===visualization===
        if args.vis:
            for icam, track_list in zip(range(1, 8 + 1), track_lists):
                # open image file
                #curr_image = plt.imread(pjoin(args.basedir, 'frames/camera{}/{}.jpg'.format(icam, lib.glob2loc(curr_frame, icam))))  # TODO
                curr_image = images[icam-1]
                plt.imshow(curr_image, extent=[0, 1920//2, 1080//2, 0])

                # plot (active) tracks
                for each_tracker in track_list:
                    #if(each_tracker.track_id==3):
                    each_tracker.plot_track(plot_past_trajectory=True, plot_heatmap=True, output_shape=(1080//2, 1920//2))
                    #break
                    #plt.gca().add_patch(patches.Rectangle((each_tracker.KF.x[0]-50, each_tracker.KF.x[2]-200),
                    #                                        100, 200, fill=False, linewidth=3, edgecolor=each_tracker.color))

                #plt.imshow(curr_heatmap,alpha=0.5,interpolation='none',cmap='hot',extent=[0,curr_image.shape[1],curr_image.shape[0],0],clim=(0, 10))
                savefig(pjoin(args.outdir, 'camera{}/res_img_{:06d}.jpg'.format(icam, curr_frame)), quality=80)
                #plt.show()
                plt.close()

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='2D tracker test.')
    parser.add_argument('--basedir', nargs='?', default='/work/breuers/dukeMTMC/',
                        help='Path to `train` folder of 2DMOT2015.')
    parser.add_argument('--outdir', nargs='?', default='/home/breuers/results/duke_mtmc/',
                        help='Where to store generated output. Only needed if `--vis` is also passed.')
    parser.add_argument('--vis', action='store_true',
                        help='Generate and save visualization of the results.')
    args = parser.parse_args()
    print(args)

    # This is all for faking the network.
    net = FakeNeuralNewsNetwork(lib.load_trainval(pjoin(args.basedir, 'ground_truth', 'trainval.mat')))

    # Prepare output dirs
    for icam in range(1, 8+1):
        makedirs(pjoin(args.outdir, 'camera{}'.format(icam)), exist_ok=True)

    tstart = time.time()
    try:
        main(net)
    except KeyboardInterrupt:
        print()

    print('FPS: {:.3f}'.format(g_frames / (time.time() - tstart)))
