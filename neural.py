from importlib import import_module

import numpy as np
import DeepFried2 as df

import lib
from lib.models import add_defaults
from fakenews import FakeNeuralNewsNetwork


class SemiFakeNews(FakeNeuralNewsNetwork):
    def __init__(self, model, weights, scale_factor, fake_dets):
        self.scale_factor = scale_factor

        self.net = add_defaults(import_module('lib.models.' + model).mknet())
        self.net.load(weights)
        self.net.evaluate()

        print("Precompiling network...", end='', flush=True)
        #self.net.forward(np.zeros((1,3) + self.net.in_shape, df.floatX))
        out = self.net.forward(np.zeros((1,3,int(1080*scale_factor),int(1920*scale_factor)), df.floatX))
        print("Done", flush=True)

        FakeNeuralNewsNetwork.__init__(self, fake_dets, fake_shape=out.shape[2:])



    # Only for the parent fake one.
    #def tick(self, curr_frame):
    #    pass  # Not needed for real network.


    # Only for the parent fake one.
    #def fake_camera(self, *fakea, **fakekw):
    #    pass  # Note needed for real network.


    def embed_crop(self, crop, *fakea, **fakekw):
        assert (crop.shape[0]*self.scale_factor, crop.shape[1]*self.scale_factor) == self.net.in_shape
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
        #return = 1/(0.01+d)


    def fix_shape(self, net_output, orig_shape, out_shape, fill_value=0):
        orig_shape = (orig_shape[0]*self.scale_factor, orig_shape[1]*self.scale_factor)

        # Scale to `out_shape` but keeping correct aspect ratio.
        h = int(self.net.scale_factor[0]/orig_shape[0]*net_output.shape[0]*out_shape[0])
        w = int(self.net.scale_factor[1]/orig_shape[1]*net_output.shape[1]*out_shape[1])
        scaled_out = lib.resize_map(net_output, (h, w))

        # Paste into the middle.
        out = np.full(out_shape, fill_value, dtype=net_output.dtype)
        dy, dx = (out.shape[0]-h)//2, (out.shape[1]-w)//2

        # TODO: Is there a better way? 'cause :-0 fails. I guess do shape[0]-dx?
        if 0 < dy and 0 < dx:
            out[dy:-dy,dx:-dx] = scaled_out
        elif dx == 0:
            out[dy:-dy,:] = scaled_out
        elif dy == 0:
            out[:,dx:-dx] = scaled_out
        else:
            print("{} = ({}-{})//2".format(dy, out.shape[0], h))
            print("{} = ({}-{})//2".format(dx, out.shape[1], w))
            assert False, "Something wrong with shape-fixing, see above!"

        return out


    # Inherited from the parent fake one
    #def personness(self, image, known_embs):
    #    # TODO: Teh big Q
    #    pass
