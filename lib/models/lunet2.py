import DeepFried2 as df
from .. import dfext


def mknet():
    kw = dict()

    net = df.Sequential(
        # -> 128x48
        df.SpatialConvolutionCUDNN(3, 64, (7,7), border='same', bias=None),
        dfext.resblock(64, **kw),
        df.PoolingCUDNN((2,2)),  # -> 64x24
        dfext.resblock(64, **kw),
        dfext.resblock(64, **kw),
        dfext.resblock(64, 96, **kw),
        df.PoolingCUDNN((2,2)),  # -> 32x12
        dfext.resblock(96, **kw),
        dfext.resblock(96, **kw),
        df.PoolingCUDNN((2,2)),  # -> 16x6
        dfext.resblock(96, **kw),
        dfext.resblock(96, **kw),
        dfext.resblock(96, 128, **kw),
        df.PoolingCUDNN((2,2)),  # -> 8x3
        dfext.resblock(128, **kw),
        dfext.resblock(128, **kw),
        df.PoolingCUDNN((2,3)),  # -> 4x1
        dfext.resblock(128, **kw),

        # Eq. to flatten + linear
        df.SpatialConvolutionCUDNN(128, 256, (4,1), bias=None),
        df.BatchNormalization(256, 0.95), df.ReLU(),

        df.StoreOut(df.SpatialConvolutionCUDNN(256, 128, (1,1)))
    )

    net.emb_mod = net[-1]
    net.in_shape = (128, 48)
    net.scale_factor = (2*2*2*2*2, 2*2*2*2*3)

    print("Net has {:.2f}M params".format(df.utils.count_params(net)/1000/1000), flush=True)
    return net
