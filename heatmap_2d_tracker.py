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
from mpl_toolkits.axes_grid1 import ImageGrid

#tracker stuff
import lib
from track import Track
from fakenews import FakeNeuralNewsNetwork
from semifake import SemiFakeNews
#from neural import RealNews


SEQ_FPS = 60.0
SEQ_DT = 1./SEQ_FPS
SEQ_SHAPE = (1080, 1920)
STATE_SHAPE = (270, 480)  # Heatmaps: (26, 58) -> (33, 60)
NUM_CAMS = 2 # which cam to consider (from 1 to NUM_CAMS), max: 8
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


#@profile
def main(net, args):
    eval_path = pjoin(args.outdir, 'results/run_{:%Y-%m-%d_%H:%M:%S}.txt'.format(datetime.datetime.now()))

    debug_dir = None
    if args.debug:
        debug_dir = pjoin(args.outdir, 'debug/run_{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now()))
        makedirs(pjoin(debug_dir, 'crops'), exist_ok=True)

    track_lists = [[] for _ in range(NUM_CAMS)]
    track_id = 1

    # ===Tracking fun begins: iterate over frames===
    # TODO: global time (duke)
    for curr_frame in range(args.t0, args.t1+1):
        print("\rFrame {}, {} matched/missed/init/total tracks, {} total seen".format(curr_frame, ', '.join(map(n_active_tracks, track_lists)), sum(map(len, track_lists))), end='', flush=True)
        net.tick(curr_frame)

        images = [plt.imread(pjoin(args.basedir, 'frames/camera{}/{}.jpg'.format(icam, lib.glob2loc(curr_frame, icam)))) for icam in range(1,8+1)]
        image_embeddings = net.embed_images(images, batch=args.large_gpu)

        for icam, track_list in zip(range(1, NUM_CAMS+1), track_lists):
            net.fake_camera(icam)


            # ===visualization===
            # First, plot what data we have before doing anything.
            if shall_vis(args, curr_frame):
                fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(20,12))
                (ax_tl, ax_tr), (ax_ml, ax_mr), (ax_bl, ax_br) = axes
                axes = axes.flatten()

                for ax in axes:
                    ax.imshow(images[icam-1], extent=[0, 1920, 1080, 0])

                # plot (active) tracks
                ax_tl.set_title('Raw Personness')
                ax_tr.set_title('Filtered Personness')
                ax_ml.set_title('Prior')
                ax_mr.set_title('All ID-specific')
                ax_bl.set_title('Posterior')
                ax_br.set_title('All Tracks')
            # ===/visualization===


            ### A) update existing tracks
            for itracker, track in enumerate(track_list):
                # ---PREDICT---
                track.track_predict()
                if shall_vis(args, curr_frame):
                    track.plot_pred_heatmap(ax_ml)

                # ---SEARCH---
                id_heatmap = net.search_person(image_embeddings[icam-1], track.embedding,
                                               fake_track_id=track.track_id)  # Unused by real net.
                id_heatmap = net.fix_shape(id_heatmap, images[icam-1].shape, STATE_SHAPE, fill_value=0)

                # ---UPDATE---
                track.track_update(id_heatmap, curr_frame, images[icam-1])

                if shall_vis(args, curr_frame):
                    track.plot_id_heatmap(ax_mr)

            ### B) get new tracks from general heatmap
            viz_per_cam_personnesses = []

            #known_embs = [track.embedding for track in track_lists[icam-1]]
            #personness = net.clear_known(image_personnesses[icam-1], image_embeddings[icam-1], known_embs=known_embs)
            #personness = net.fix_shape(personness, images[icam-1].shape, STATE_SHAPE, fill_value=0)
            #viz_per_cam_personnesses.append(personness)

            # B.1) COMMENT IN FOR SEMI-FAKE
            # TODO: Make semi-fake by generating heatmap and clearing out known_embs
            for new_heatmap, new_id in net.personness(images[icam-1], known_embs=None):
                # TODO: get correct track_id (loop heatmap, instead of function call?# )
                # TODO: get id_heatmap of that guy for init_heatmap
                new_heatmap = net.fix_shape(new_heatmap, images[icam-1].shape, STATE_SHAPE, fill_value=0)
                init_pose = lib.argmax2d_xy(new_heatmap)
                new_track = Track(net.embed_crops, SEQ_DT,
                                  curr_frame, init_pose, images[icam-1], track_id=new_id,
                                  state_shape=STATE_SHAPE, output_shape=SEQ_SHAPE,
                                  person_matching_threshold=0.001,
                                  debug_out_dir=debug_dir)
                new_track.init_heatmap(new_heatmap)
                track_lists[icam-1].append(new_track)

            # B.2) REAL NEWS
            # TODO: Missing non-max suppression
            # for y_idx, x_idx in zip(*np.where(personness>1.5)):
            #     init_pose = [y_idx, x_idx]
            #     new_track = Track(net.embed_crop, SEQ_DT,
            #                       curr_frame, init_pose, images[icam-1], track_id=track_id,
            #                       state_shape=STATE_SHAPE, output_shape=SEQ_SHAPE,
            #                       person_matching_threshold=0.001,
            #                       debug_out_dir=debug_dir)

            #     # Embed around the initial pose and compute an initial heatmap.
            #     id_heatmap = net.search_person(image_embeddings[icam-1], new_track.embedding)
            #     id_heatmap = net.fix_shape(id_heatmap, images[icam-1].shape, STATE_SHAPE, fill_value=0)
            #     new_track.init_heatmap(id_heatmap)
            #     track_id += 1
            #     track_list.append(new_track)

            if shall_vis(args, curr_frame):
                track.plot_pos_heatmap(ax_bl)
                track.plot_track(ax_br, plot_past_trajectory=True)

                for ax in axes:
                    # TODO: Flex
                    ax.set_adjustable('box-forced')
                    ax.set_xlim(0, 1920)
                    ax.set_ylim(1080, 0)
                    fig.savefig(pjoin(args.outdir, 'camera{}/res_img_{:06d}.jpg'.format(icam, curr_frame)),
                                quality=80, bbox_inches='tight', pad_inches=0.2)
                    plt.close()

            ### C) further track-management
            # delete tracks marked as 'deleted' in this tracking cycle #TODO: manage in other list for re-id
            #track_list = [i for i in track_list if i.status != 'deleted']


        # ==evaluation===
        with open(eval_path, 'a') as eval_file:
            for icam, track_list in zip(range(1, 8 + 1), track_lists):
                for track in track_list:
                    track_eval_line = track.get_track_eval_line(cid=icam, frame=curr_frame)
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='2D tracker test.')
    parser.add_argument('--basedir', nargs='?', default='/work/breuers/dukeMTMC/',
                        help='Path to `train` folder of 2DMOT2015.')
    parser.add_argument('--outdir', nargs='?', default='/home/breuers/results/duke_mtmc/',
                        help='Where to store generated output. Only needed if `--vis` is also passed.')
    parser.add_argument('--model', default='lunet2',
                        help='Name of the model to load. Corresponds to module names in lib/models. Or `fake`')
    parser.add_argument('--weights', default='/work/breuers/dukeMTMC/models/lunet2-final.pkl',
                        help='Name of the weights to load for the model (path to .pkl file).')
    parser.add_argument('--t0', default=49700, type=int,
                        help='Time of first frame.')
    parser.add_argument('--t1', default=227540, type=int,
                        help='Time of last frame, inclusive.')
    parser.add_argument('--large_gpu', action='store_true',
                        help='Large GPU can forward more at once.')
    parser.add_argument('--vis', default=0, type=int,
                        help='Generate and save visualization of the results, every X frame.')
    parser.add_argument('--debug', action='store_true',
                        help='Generate extra many debugging outputs (in outdir).')
    args = parser.parse_args()
    print(args)

    # This is all for faking the network.
    if args.model == 'fake':
        net = FakeNeuralNewsNetwork(lib.load_trainval(pjoin(args.basedir, 'ground_truth', 'trainval.mat'), time_range=[args.t0, args.t1]))
    else:
        #net = RealNews(
        net = SemiFakeNews(
            model=args.model,
            weights=args.weights,
            input_scale_factor=0.5,
            fake_dets=lib.load_trainval(pjoin(args.basedir, 'ground_truth', 'trainval.mat'), time_range=[args.t0, args.t1])
        )

    # Prepare output dirs
    for icam in range(1, 8+1):
        makedirs(pjoin(args.outdir, 'camera{}'.format(icam)), exist_ok=True)
    makedirs(pjoin(args.outdir, 'results'), exist_ok=True)

    tstart = time.time()
    try:
        main(net, args)
    except KeyboardInterrupt:
        print()

    print('FPS: {:.3f}'.format(g_frames / (time.time() - tstart)))
