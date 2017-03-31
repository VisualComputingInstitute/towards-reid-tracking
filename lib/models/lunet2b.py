import DeepFried2 as df
from .. import dfext


def mknet(mkbn=lambda chan: df.BatchNormalization(chan, 0.95), avg=True, initlast=df.init.xavier()):
    kw = dict(mkbn=mkbn)

    net = df.Sequential(
        # -> 128x48
        df.SpatialConvolutionCUDNN(3, 64, (7,7), border='same', bias=None),
        dfext.resblock2(64, **kw),
        df.PoolingCUDNN((2,2)),  # -> 64x24
        dfext.resblock2(64, **kw),
        dfext.resblock2(64, **kw),
        dfext.resblock2(64, 96, **kw),
        df.PoolingCUDNN((2,2)),  # -> 32x12
        dfext.resblock2(96, **kw),
        dfext.resblock2(96, **kw),
        dfext.resblock2(96, 128, **kw),
        df.PoolingCUDNN((2,2)),  # -> 16x6
        dfext.resblock2(128, **kw),
        dfext.resblock2(128, **kw),
        dfext.resblock2(128, 192, **kw),
        df.PoolingCUDNN((2,2)),  # -> 8x3
        dfext.resblock2(192, **kw),
        dfext.resblock2(192, **kw),
    )

    if avg:
        net.add(dfext.resblock2(192, 256, **kw))
        net.add(mkbn(256))
        net.add(df.ReLU())
        net.add(df.PoolingCUDNN((8,3), mode='avg'))  # -> 1x1

        net.add(df.SpatialConvolutionCUDNN(256, 192, (1,1), bias=None, init=df.init.prelu()))
        net.add(mkbn(192))
        net.add(df.ReLU())
    else:
        net.add(df.PoolingCUDNN((2,3)))  # -> 4x1
        net.add(dfext.resblock2(128, **kw))
        net.add(mkbn(128))
        net.add(df.ReLU())
        net.add(df.SpatialConvolutionCUDNN(128, 256, (4,1), bias=None, init=df.init.prelu()))
        net.add(mkbn(256))
        net.add(df.ReLU())

    net.add(df.StoreOut(df.SpatialConvolutionCUDNN(256, 128, (1,1), init=initlast)))

    net.emb_mod = net[-1]
    net.in_shape = (128, 48)
    net.scale_factor = None # TODO (2*2*2*2*2, 2*2*2*2*3)

    print("Net has {:.2f}M params".format(df.utils.count_params(net)/1000/1000), flush=True)
    return net
