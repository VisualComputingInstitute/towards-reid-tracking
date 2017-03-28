from importlib import import_module

import numpy as np
import DeepFried2 as df

import lib
from lib.models import add_defaults
from fakenews import FakeNeuralNewsNetwork


class SemiFakeNews(FakeNeuralNewsNetwork):
    def __init__(self, model, weights, scale_factor, fake_dets):
        FakeNeuralNewsNetwork.__init__(self, fake_dets)

        self.scale_factor = scale_factor

        self.net = add_defaults(import_module('lib.models.' + model).mknet())
        self.net.load(weights)
        self.net.evaluate()

        print("Precompiling network...", end='', flush=True)
        self.net.forward(np.zeros((1,3) + self.net.in_shape, df.floatX))
        print("Done", flush=True)


    # Only for the parent fake one.
    #def tick(self, curr_frame):
    #    pass  # Not needed for real network.


    # Only for the parent fake one.
    #def fake_camera(self, *fakea, **fakekw):
    #    pass  # Note needed for real network.


    def embed_crop(self, crop, *fakea, **fakekw):
        X = lib.img2df(crop, shape=self.net.in_shape)
        return self.net.forward(X[None])[0,:,0,0]


    def embed_image(self, image):
        # TODO: resize? multi-scale?
        H, W, _ = image.shape
        X = lib.img2df(image, shape=(int(H*self.scale_factor), int(W*self.scale_factor)))
        return self.net.forward(X[None])[0]


    def search_person(self, img_embs, person_emb, *fakea, **fakekw):
        # compute distance between embeddings and person's embedding.
        d = np.sqrt(np.sum((img_embs - person_emb[:,None,None])**2, axis=0))

        # Convert distance to probability.
        # TODO: Might be better to fit a sigmoid or something.
        return lib.softmin(d)
        #return 1/(0.01+d)


    # Inherited from the parent fake one
    #def personness(self, image, known_embs):
    #    # TODO: Teh big Q
    #    pass
