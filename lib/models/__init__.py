def add_defaults(net):
    if not hasattr(net, 'load'):
        net.load = lambda fname: net.__setstate__(pickle.load(open(fname, 'rb')))
    return net
