#!/usr/bin/env python3
import argparse
from importlib import import_module
from os.path import splitext, join as pjoin

import cv2
import numpy as np
import h5py

import lib
from lib.models import add_defaults


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embed many images.')
    parser.add_argument('--basedir', default='.',
                        help='Path to the folder containing all images.')
    parser.add_argument('--outfile', default='embeddings.h5',
                        help='Name of the output hdf5 file in which to store the embeddings.')
    parser.add_argument('--model', default='lunet2',
                        help='Name of the model to load. Corresponds to module names in lib/models. Or `fake`')
    parser.add_argument('--weights', default='/work/breuers/dukeMTMC/models/lunet2-final.pkl',
                        help='Name of the weights to load for the model (path to .pkl file).')
    parser.add_argument('--scale', default=1.0, type=float,
                        help='Scale factor to scale images before embedding them.')
    parser.add_argument('--t0', type=int)
    parser.add_argument('--t1', type=int)
    args = parser.parse_args()
    print(args)


    mod = import_module('lib.models.' + args.model)
    net = add_defaults(mod.mknet())

    try:
        net.load(args.weights)
    except ValueError:
        print("!!!!!!!THE WEIGHTS YOU LOADED DON'T BELONG TO THE MODEL YOU'RE USING!!!!!!")
        raise

    # Shares the weights, just replaces the avg-pooling layer.
    net_hires = mod.hires_shared_twin(net)
    net_hires.evaluate()

    if args.t0 is None or args.t1 is None:
        all_files = sane_listdir(args.basedir, sortkey=lambda f: int(splitext(f)[0]))
    else:
        all_files = ['{}.jpg'.format(i) for i in range(args.t0, args.t1+1)]

    print("Precompiling network...", end='', flush=True)
    img = lib.imread(pjoin(args.basedir, all_files[0]))
    img = lib.img2df(img, lib.scale_shape(img.shape, args.scale))
    out = net_hires.forward(img[None])
    print(" Done", flush=True)

    with h5py.File(args.outfile, 'w') as f_out:
        ds = f_out.create_dataset('embs', shape=(len(all_files),) + out.shape[1:], dtype=out.dtype)
        for i, fname in enumerate(all_files):
            print("\r{} ({}/{})".format(fname, i, len(all_files)), end='', flush=True)

            img = lib.imread(pjoin(args.basedir, fname))
            img = lib.img2df(img, lib.scale_shape(img.shape, args.scale))
            ds[i] = net_hires.forward(img[None])

            if i % 100 == 0:
                f_out.flush()

    print(" Done")
