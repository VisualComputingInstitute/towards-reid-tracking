import pickle


def _raise_fn(ex):
    def fn(*a, **kw):
        raise ex
    return fn


def add_defaults(net):
    #net.embed = lambda f, idxs: np.concatenate([net.forward(net.raw2df(get(f['X'], ib))) for ib in batched(128, idxs)])
    if not hasattr(net, 'load'):
        net.load = lambda fname: net.__setstate__(pickle.load(open(fname, 'rb')))
    return net
