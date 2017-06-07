#!/usr/bin/env python
# -*- coding: utf-8 -*-.

from __future__ import print_function
from __future__ import division

import argparse
from os.path import join as pjoin
from os import makedirs
import time, datetime

# the usual suspects
import h5py
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
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
STATE_SHAPE = (135, 240)  # Heatmaps: (26, 58) -> (33, 60)
STATE_PADDING = ((5,5), (10,10))  # state shape is this much larger on the sides, see np.pad.


g_frames = 0  # Global counter for correct FPS in all cases


try:
    profile
except NameError:
    def profile(f):
        return f


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


@lib.lru_cache(maxsize=16)  # In theory 1 is enough here, but whatever =)
def get_image(basedir, icam, frame):
    #framedir = 'frames-0.5' if SCALE_FACTOR == 0.5 else 'frames'
    # TODO: Use basedir again, from args.
    return plt.imread(pjoin('/work3/beyer/', 'frames-0.5', 'camera{}/{}.jpg'.format(icam, lib.glob2loc(frame, icam))))


@profile
def main(net, args):
    eval_path = pjoin(args.outdir, 'results/run_{:%Y-%m-%d_%H:%M:%S}.txt'.format(datetime.datetime.now()))

    debug_dir = None
    if args.debug:
        debug_dir = pjoin(args.outdir, 'debug/run_{:%Y-%m-%d_%H:%M:%S}'.format(datetime.datetime.now()))
        makedirs(pjoin(debug_dir, 'crops'), exist_ok=True)

    track_lists = [[] for _ in args.cams]
    track_id = 1

    # Open embedding cache
    if args.embcache is not None:
        embs_caches = [h5py.File(args.embcache.format(icam), 'r')['embs'] for icam in args.cams]
    else:
        embs_caches = [None]*len(args.cams)

    # ===Tracking fun begins: iterate over frames===
    # TODO: global time (duke)
    for curr_frame in range(args.t0, args.t1+1):
        print("\rFrame {}, {} matched/missed/init/total tracks, {} total seen".format(curr_frame, ', '.join(map(n_active_tracks, track_lists)), sum(map(len, track_lists))), end='', flush=True)
        net.tick(curr_frame)

        for icam, track_list, embs_cache in zip(args.cams, track_lists, embs_caches):
            net.fake_camera(icam)

            image_getter = lambda: get_image(args.basedir, icam, curr_frame)

            # Either embed the image, or load embedding from cache.
            if embs_cache is not None:
                image_embedding = np.array(embs_cache[curr_frame-127720])  # That's where the cache starts!
            else:
                image_embedding = net.embed_images([image_getter()])[0]


            # ===visualization===
            # First, plot what data we have before doing anything.
            if shall_vis(args, curr_frame):
                #fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(20,12))
                #(ax_tl, ax_tr), (ax_ml, ax_mr), (ax_bl, ax_br) = axes
                fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(20,12))
                (ax_ml, ax_mr), (ax_bl, ax_br) = axes
                axes = axes.flatten()

                for ax in axes:
                    ax.imshow(image_getter(), extent=[0, SEQ_SHAPE[1], SEQ_SHAPE[0], 0])

                # plot (active) tracks
                #ax_tl.set_title('Raw Personness')
                #ax_tr.set_title('Filtered Personness')
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
                id_distmap = net.search_person(image_embedding, track.embedding, T=1,
                                               fake_track_id=track.track_id)  # Unused by real net.
                # FIXME: should be image.shape, or at least use scale-factor.
                id_distmap = net.fix_shape(id_distmap, (1080//2, 1920//2), STATE_SHAPE, fill_value=1/np.prod(STATE_SHAPE))
                id_heatmap = lib.softmin(id_distmap, T=1)
                #id_heatmap /= np.sum(id_heatmap)

                # ---UPDATE---
                track.track_update(id_heatmap, id_distmap, curr_frame, image_getter)

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
            # TODO: use image instead of None for real one here
            for (new_heatmap, init_pose), new_id in net.personness(None, known_embs=None, return_pose=True):
                # TODO: get correct track_id (loop heatmap, instead of function call?# )
                # TODO: get id_heatmap of that guy for init_heatmap
                # Don't fix shape yet, cuz we don't emulate the avg-pool shape screw-up.
                #new_heatmap = net.fix_shape(new_heatmap, images[icam-1].shape, STATE_SHAPE, fill_value=0)
                #init_pose = lib.argmax2d_xy(new_heatmap)
                new_track = Track(net.embed_crops,
                                  curr_frame, init_pose, image_getter(), track_id=new_id,
                                  state_shape=STATE_SHAPE, state_pad=STATE_PADDING, output_shape=SEQ_SHAPE,
                                  dist_thresh=args.dist_thresh, entropy_thresh=args.ent_thresh,
                                  unmiss_thresh=args.unmiss_thresh, delete_thresh=args.delete_thresh,
                                  maxlife=args.maxlife, tp_hack=args.tp_hack,
                                  debug_out_dir=debug_dir)
                new_track.init_heatmap(new_heatmap)
                #new_track.init_heatmap(np.full(STATE_SHAPE, 1/np.prod(STATE_SHAPE)))
                track_list.append(new_track)

            # B.2) REAL NEWS
            # TODO: Missing non-max suppression
            # for y_idx, x_idx in zip(*np.where(personness>1.5)):
            #     init_pose = [y_idx, x_idx]
            #     new_track = Track(net.embed_crop, SEQ_DT,
            #                       curr_frame, init_pose, images[icam-1], track_id=track_id,
            #                       state_shape=STATE_SHAPE, output_shape=SEQ_SHAPE,
            #                       debug_out_dir=debug_dir)

            #     # Embed around the initial pose and compute an initial heatmap.
            #     id_heatmap = net.search_person(image_embeddings[icam-1], new_track.embedding)
            #     id_heatmap = net.fix_shape(id_heatmap, images[icam-1].shape, STATE_SHAPE, fill_value=0)
            #     new_track.init_heatmap(id_heatmap)
            #     track_id += 1
            #     track_list.append(new_track)

            if shall_vis(args, curr_frame):
                for track in track_list:
                    track.plot_pos_heatmap(ax_bl)
                    track.plot_track(ax_br, plot_past_trajectory=True, time_scale=args.vis)

                for ax in axes:
                    # TODO: Flex
                    ax.set_adjustable('box-forced')
                    ax.set_xlim(0, SEQ_SHAPE[1])
                    ax.set_ylim(SEQ_SHAPE[0], 0)
                    fig.savefig(pjoin(args.outdir, 'camera{}/res_img_{:06d}.jpg'.format(icam, curr_frame)),
                                quality=80, bbox_inches='tight', pad_inches=0.2)
                    plt.close()

            ### C) further track-management
            # delete tracks marked as 'deleted' in this tracking cycle #TODO: manage in other list for re-id
            track_list[:] = [i for i in track_list if i.status != 'deleted']


        # ==evaluation===
        with open(eval_path, 'a') as eval_file:
            for icam, track_list in zip(args.cams, track_lists):
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
    parser.add_argument('--model', default='lunet2c',
                        help='Name of the model to load. Corresponds to module names in lib/models. Or `fake`')
    parser.add_argument('--weights', default='/work/breuers/dukeMTMC/models/lunet2c-noscale-nobg-2to32-aug.pkl',
                        help='Name of the weights to load for the model (path to .pkl file).')
    parser.add_argument('--t0', default=127720, type=int,
                        help='Time of first frame.')
    parser.add_argument('--t1', default=187540, type=int,
                        help='Time of last frame, inclusive.')
    parser.add_argument('--large_gpu', action='store_true',
                        help='Large GPU can forward more at once.')
    parser.add_argument('--vis', default=0, type=int,
                        help='Generate and save visualization of the results, every X frame.')
    parser.add_argument('--debug', action='store_true',
                        help='Generate extra many debugging outputs (in outdir).')
    parser.add_argument('--cams', default='1,2,3,4,5,6,7,8',
                        help='Array of cameras numbers (1-8) to consider.')
    parser.add_argument('--embcache',
                        help='Optional path to embeddings-cache file for speeding things up. Put a `{}` as placeholder for camera-number.')
    parser.add_argument('--dist_thresh', default=7, type=float,
                        help='Distance threshold to evaluate measurment certainty.')
    parser.add_argument('--ent_thresh', default=0.1, type=float,
                        help='Entropy threshold to evaluate measurment certainty.')
    parser.add_argument('--maxlife', type=int)
    parser.add_argument('--tp_hack', type=float)
    parser.add_argument('--unmiss_thresh', type=int, default=2)
    parser.add_argument('--delete_thresh', type=int, default=90)
    args = parser.parse_args()
    args.cams = eval('[' + args.cams + ']')
    print(args)

    # This is all for faking the network.
    if args.model == 'fake':
        net = FakeNeuralNewsNetwork(lib.load_trainval(pjoin(args.basedir, 'ground_truth', 'trainval.mat'), time_range=[args.t0, args.t1]))
    else:
        #net = RealNews(
        net = SemiFakeNews(
            model=args.model,
            weights=args.weights,
            input_scale_factor=1.0,
            fake_dets=lib.load_trainval(pjoin(args.basedir, 'ground_truth', 'trainval.mat'), time_range=[args.t0, args.t1]),
            fake_shape=STATE_SHAPE,
        )

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
