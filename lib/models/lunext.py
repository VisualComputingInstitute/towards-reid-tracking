import DeepFried2 as df
from .. import dfext


def mknet():
    net = df.Sequential(
        # -> 128x48
        df.SpatialConvolutionCUDNN(3, 128, (7,7), border='same', bias=None, init=df.init.prelu()),
        df.BatchNormalization(128, 0.95), df.ReLU(),

        dfext.nextblock_b(128, cardin=16, chan_mid=4),
        df.PoolingCUDNN((2,2)),  # -> 64x24
        dfext.nextblock_b(128, cardin=16, chan_mid=4),
        dfext.nextblock_b(128, cardin=16, chan_mid=4),
        dfext.nextblock_b(128, cardin=16, chan_mid=4, chan_out=256),
        df.PoolingCUDNN((2,2)),  # -> 32x12
        dfext.nextblock_b(256, cardin=16, chan_mid=8),
        dfext.nextblock_b(256, cardin=16, chan_mid=8),
        df.PoolingCUDNN((2,2)),  # -> 16x6
        dfext.nextblock_b(256, cardin=16, chan_mid=8),
        dfext.nextblock_b(256, cardin=16, chan_mid=8),
        dfext.nextblock_b(256, cardin=16, chan_mid=8, chan_out=512),
        df.PoolingCUDNN((2,2)),  # -> 8x3
        dfext.nextblock_b(512, cardin=16, chan_mid=16),
        dfext.nextblock_b(512, cardin=16, chan_mid=16),
        df.PoolingCUDNN((8,3), mode='avg'),
        df.SpatialConvolutionCUDNN(512, 256, (1,1), bias=None, init=df.init.prelu()),
        df.BatchNormalization(256, 0.95), df.ReLU(),
        df.StoreOut(df.SpatialConvolutionCUDNN(256, 128, (1,1)))
    )

    net.emb_mod = net[-1]
    net.in_shape = (128, 48)
    net.scale_factor = None  # TODO

    print("Net has {:.2f}M params".format(df.utils.count_params(net)/1000/1000), flush=True)
    return net
