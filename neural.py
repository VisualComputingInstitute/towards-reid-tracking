from importlib import import_module

import numpy as np
import DeepFried2 as df

import lib
from lib.models import add_defaults
from fakenews import FakeNeuralNewsNetwork


class RealNews:
    def __init__(self, model, weights, scale_factor):
        self.scale_factor = scale_factor

        mod = import_module('lib.models.' + model)
        self.net = add_defaults(mod.add_piou(mod.mknet()))

        try:
            self.net.load(weights)
        except ValueError:
            print("!!!!!!!THE WEIGHTS YOU LOADED DON'T BELONG TO THE MODEL YOU'RE USING!!!!!!")
            raise

        self.net.evaluate()

        print("Precompiling network...", end='', flush=True)
        #self.net.forward(np.zeros((1,3) + self.net.in_shape, df.floatX))
        self.net.forward(np.zeros((1,3,int(1080*scale_factor),int(1920*scale_factor)), df.floatX))
        print("Done", flush=True)


    def tick(self, curr_frame):
        pass  # Not needed for real network.


    def fake_camera(self, *fakea, **fakekw):
        pass  # Note needed for real network.


    def embed_crop(self, crop, *fakea, **fakekw):
        assert (crop.shape[0]*self.scale_factor, crop.shape[1]*self.scale_factor) == self.net.in_shape
        X = lib.img2df(crop, shape=self.net.in_shape)
        return self.net.embs_from_out(self.net.forward(X[None]))[0,:,0,0]


    def embed_image(self, image):
        print("You better use `embed_and_personness_multi`, you lazy bastard")
        return self.embed_and_personness_multi([image])[0][0]


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


    def personness(self, image, known_embs):
        raise NotImplementedError("TODO. Use `embed_and_personness_multi` instead, don't be wasteful!")


    def embed_and_personness_multi(self, images, batch=True):
        H, W, _ = images[0].shape

        if batch:
            out = self.net.forward(np.array([lib.img2df(img, shape=(int(H*self.scale_factor), int(W*self.scale_factor))) for img in images]))
            return self.net.embs_from_out(out), self.net.ious_from_out(out)
        else:
            embs, ious = [], []
            for img in images:
                out = self.net.forward(lib.img2df(img, shape=(int(H * self.scale_factor), int(W * self.scale_factor)))[None])
                embs.append(self.net.embs_from_out(out)[0])
                ious.append(self.net.ious_from_out(out)[0])
            return np.array(embs), np.array(ious)


    def clear_known(self, image_personness, image_embs, known_embs):
        p_iou = np.array(image_personness)
        for emb in known_embs:
            p_emb = self.search_person(image_embs, emb)
            p_iou *= 1-p_emb
        return p_iou


class SemiFakeNews:
    def __init__(self, model, weights, scale_factor, fake_dets):
        self.real = RealNews(model, weights, scale_factor)

        out = self.real.embed_image(np.zeros((3,1080,1920), df.floatX))
        self.fake = FakeNeuralNewsNetwork(fake_dets, fake_shape=out.shape[2:])


    def tick(self, *a, **kw):
        self.real.tick(*a, **kw)
        self.fake.tick(*a, **kw)


    def fake_camera(self, *a, **kw):
        self.real.fake_camera(*a, **kw)
        self.fake.fake_camera(*a, **kw)


    def embed_crop(self, crop, *fakea, **fakekw):
        return self.real.embed_crop(crop)


    def embed_image(self, image):
        return self.real.embed_image(image)


    def search_person(self, img_embs, person_emb, *fakea, **fakekw):
        return self.real.search_person(img_embs, person_emb)


    def fix_shape(self, net_output, orig_shape, out_shape, fill_value=0):
        return self.real.fix_shape(net_output, orig_shape, out_shape)


    def personness(self, image, known_embs):
        return self.fake.personness(image, known_embs)
