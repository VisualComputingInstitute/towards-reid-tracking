from importlib import import_module

import numpy as np
import DeepFried2 as df
from scipy.spatial.distance import cdist

from lbtoolbox.util import batched

import lib
from lib.models import add_defaults
from fakenews import FakeNeuralNewsNetwork


class SemiFakeNews:
    def __init__(self, model, weights, input_scale_factor, fake_shape, fake_dets, debug_skip_full_image=False):
        self.input_scale_factor = input_scale_factor

        mod = import_module('lib.models.' + model)

        self.net = mod.mknet()
        add_defaults(self.net)

        try:
            self.net.load(weights)
        except ValueError:
            print("!!!!!!!THE WEIGHTS YOU LOADED DON'T BELONG TO THE MODEL YOU'RE USING!!!!!!")
            raise

        # Shares the weights, just replaces the avg-pooling layer.
        self.net_hires = mod.hires_shared_twin(self.net)
        add_defaults(self.net_hires)

        self.net.evaluate()
        self.net_hires.evaluate()

        print("Precompiling network... 1/2", end='', flush=True)
        self.net.forward(np.zeros((1,3) + self.net.in_shape, df.floatX))
        print("\rPrecompiling network... 2/2", end='', flush=True)
        if not (debug_skip_full_image and fake_dets is None):
            out = self.net_hires.forward(np.zeros((1,3) + self._scale_input_shape((1080,1920)), df.floatX))
        print(" Done", flush=True)

        #fake_shape = out.shape[2:]  # We didn't fake the avg-pool effect yet, so don't!
        self.fake = FakeNeuralNewsNetwork(fake_dets, shape=fake_shape) if fake_dets is not None else None


    def _scale_input_shape(self, shape):
        return lib.scale_shape(shape, self.input_scale_factor)


    # Only for fake
    def tick(self, *a, **kw):
        if self.fake is not None:
            self.fake.tick(*a, **kw)


    # Only for fake
    def fake_camera(self, *a, **kw):
        if self.fake is not None:
            self.fake.fake_camera(*a, **kw)


    def embed_crops(self, crops, *fakea, batchsize=32, **fakekw):
        assert all(self._scale_input_shape(crop.shape) == self.net.in_shape for crop in crops)

        X = np.array([lib.img2df(crop, shape=self.net.in_shape) for crop in crops])
        out = np.concatenate([self.net.forward(Xb) for Xb in batched(batchsize, X)])
        return out[:,:,0,0]  # Output is Dx1x1


    def embeddings_cdist(self, embsA, embsB):
        return cdist(embsA, embsB)


    def embed_images(self, images, batch=True):
        # TODO: batch=False
        X = np.array([lib.img2df(img, shape=self._scale_input_shape(img.shape)) for img in images])
        return self.net_hires.forward(X)


    def search_person(self, img_embs, person_emb, *fakea, **fakekw):
        # compute distance between embeddings and person's embedding.
        d = np.sqrt(np.sum((img_embs - person_emb[:,None,None])**2, axis=0))

        # Convert distance to probability.
        return lib.softmin(d)  # TODO: Might be better to fit a sigmoid or something.
        #return = 1/(0.01+d)


    def scale_map(self, x):
        return lib.resize_map(x, (int(x.shape[0]*self.net.scale_factor[0]), int(x.shape[1]*self.net.scale_factor[1])))


    def fix_shape(self, net_output, orig_shape, out_shape, fill_value=0):
        orig_shape = self._scale_input_shape(orig_shape)

        # Scale to `out_shape` but keeping correct aspect ratio.
        h = net_output.shape[0]*self.net.scale_factor[0]  /orig_shape[0]*out_shape[0]
        w = net_output.shape[1]*self.net.scale_factor[1]  /orig_shape[1]*out_shape[1]

        return lib.paste_into_middle_2d(lib.resize_map(net_output, (int(h), int(w))), out_shape, fill_value)


    # THIS IS THE ONLY THING FAKE :(
    # TODO: Make semi-fake, by clearing out known_embs.
    def personness(self, image, known_embs, return_pose=False):
        assert self.fake is not None, "The world doesn't work that way my friend!"
        return self.fake.personness(image, known_embs, return_pose)
