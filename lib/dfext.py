import DeepFried2 as df


def resblock(chan_in, chan_out=None, chan_mid=None, stride=1,
             mkbn=lambda chan: df.BatchNormalization(chan, 0.95),
             mknl=lambda: df.ReLU()):
    chan_out = chan_out or chan_in
    chan_mid = chan_mid or chan_in
    return df.Sequential(
        df.RepeatInput(
            df.Sequential(
                mkbn(chan_in), mknl(),
                df.SpatialConvolutionCUDNN(chan_in, chan_mid, (3,3), border='same', stride=stride, init=df.init.prelu(), bias=False),
                mkbn(chan_mid), mknl(),
                df.SpatialConvolutionCUDNN(chan_mid, chan_out, (3,3), border='same', init=df.init.prelu()),
            ),
            df.Identity() if chan_in == chan_out else df.SpatialConvolutionCUDNN(chan_in, chan_out, (1,1), stride=stride)
        ),
        df.zoo.resnet.Add()
    )


def resblock2(chan_in, chan_out=None, chan_mid=None, stride=1,
              mkbn=lambda chan: df.BatchNormalization(chan, 0.95),
              mknl=lambda: df.ReLU()):
    chan_out = chan_out or chan_in
    chan_mid = chan_mid or chan_in

    identity_or_projection = df.Identity()
    if chan_in != chan_out:
        identity_or_projection = df.Sequential(
            mkbn(chan_in), mknl(),
            df.SpatialConvolutionCUDNN(chan_in, chan_out, (1,1), stride=stride, init=df.init.prelu()),
        )

    return df.Sequential(
        df.RepeatInput(
            df.Sequential(
                mkbn(chan_in), mknl(),
                df.SpatialConvolutionCUDNN(chan_in, chan_mid, (3,3), border='same', stride=stride, init=df.init.prelu(), bias=False),
                mkbn(chan_mid), mknl(),
                df.SpatialConvolutionCUDNN(chan_mid, chan_out, (3,3), border='same', init=df.init.prelu()),
            ),
            identity_or_projection,
        ),
        df.zoo.resnet.Add()
    )


def resblock_bottle(chan_in, chan_out=None, chan_mid=None, stride=1,
                    mkbn=lambda chan: df.BatchNormalization(chan, 0.95),
                    mknl=lambda: df.ReLU()):
    chan_out = chan_out or chan_in
    chan_mid = chan_mid or chan_out//4
    return df.Sequential(
        df.RepeatInput(
            df.Sequential(
                mkbn(chan_in), mknl(),
                df.SpatialConvolutionCUDNN(chan_in, chan_mid, (1,1), stride=stride, init=df.init.prelu(), bias=False),

                mkbn(chan_mid), mknl(),
                df.SpatialConvolutionCUDNN(chan_mid, chan_mid, (3,3), border='same', init=df.init.prelu(), bias=False),

                mkbn(chan_mid), mknl(),
                df.SpatialConvolutionCUDNN(chan_mid, chan_out, (1,1), init=df.init.prelu()),
            ),
            df.Identity() if chan_in == chan_out else df.SpatialConvolutionCUDNN(chan_in, chan_out, (1,1), stride=stride)
        ),
        df.zoo.resnet.Add()
    )


def resblock_bottle2(chan_in, chan_out=None, chan_mid=None, stride=1,
                     mkbn=lambda chan: df.BatchNormalization(chan, 0.95),
                     mknl=lambda: df.ReLU()):
    chan_out = chan_out or chan_in
    chan_mid = chan_mid or chan_out//4

    identity_or_projection = df.Identity()
    if chan_in != chan_out:
        identity_or_projection = df.Sequential(
            mkbn(chan_in), mknl(),
            df.SpatialConvolutionCUDNN(chan_in, chan_out, (1,1), stride=stride, init=df.init.prelu()),
        )

    return df.Sequential(
        df.RepeatInput(
            df.Sequential(
                mkbn(chan_in), mknl(),
                df.SpatialConvolutionCUDNN(chan_in, chan_mid, (1,1), init=df.init.prelu(), bias=False),

                mkbn(chan_mid), mknl(),
                df.SpatialConvolutionCUDNN(chan_mid, chan_mid, (3,3), stride=stride, border='same', init=df.init.prelu(), bias=False),

                mkbn(chan_mid), mknl(),
                df.SpatialConvolutionCUDNN(chan_mid, chan_out, (1,1), init=df.init.prelu()),
            ),
            identity_or_projection,
        ),
        df.zoo.resnet.Add()
    )


def repeat_apply_merge(modules, merger, *tail):
    return df.Sequential(df.RepeatInput(*modules), merger, *tail)


def nextblock_a(chan_in, cardin, chan_out=None, chan_mid=None, stride=1,
                mkbn=lambda chan: df.BatchNormalization(chan, 0.95),
                mknl=lambda: df.ReLU()):
    chan_out = chan_out or chan_in
    chan_mid = chan_mid or chan_out//cardin//2

    identity_or_projection = df.Identity()
    if chan_in != chan_out:
        identity_or_projection = df.Sequential(
            df.SpatialConvolutionCUDNN(chan_in, chan_out, (1,1), stride=stride, init=df.init.prelu()),
            mkbn(chan_out),
        )

    return repeat_apply_merge([
        repeat_apply_merge([
            df.Sequential(
                df.SpatialConvolutionCUDNN(chan_in, chan_mid, (1,1), init=df.init.prelu(), bias=False),
                mkbn(chan_mid), mknl(),

                df.SpatialConvolutionCUDNN(chan_mid, chan_mid, (3,3), init=df.init.prelu(), bias=False,
                                           stride=stride, border='same'),
                mkbn(chan_mid), mknl(),

                df.SpatialConvolutionCUDNN(chan_mid, chan_out, (1,1), init=df.init.prelu(), bias=False),
            ) for _ in range(cardin)
        ], df.zoo.resnet.Add(), mkbn(chan_out)),
        identity_or_projection
    ], df.zoo.resnet.Add(), mknl())


def nextblock_b(chan_in, cardin, chan_out=None, chan_mid=None, stride=1,
                mkbn=lambda chan: df.BatchNormalization(chan, 0.95),
                mknl=lambda: df.ReLU()):
    chan_out = chan_out or chan_in
    chan_mid = chan_mid or chan_out//cardin//2

    identity_or_projection = df.Identity()
    if chan_in != chan_out:
        identity_or_projection = df.Sequential(
            df.SpatialConvolutionCUDNN(chan_in, chan_out, (1,1), stride=stride, init=df.init.prelu()),
            mkbn(chan_out),
        )

    return repeat_apply_merge([
        repeat_apply_merge([
            df.Sequential(
                df.SpatialConvolutionCUDNN(chan_in, chan_mid, (1,1), init=df.init.prelu(), bias=False),
                mkbn(chan_mid), mknl(),

                df.SpatialConvolutionCUDNN(chan_mid, chan_mid, (3,3), init=df.init.prelu(), bias=False,
                                           stride=stride, border='same'),
                mkbn(chan_mid), mknl(),
            ) for _ in range(cardin)
            ],
            df.Concat(),
            df.SpatialConvolutionCUDNN(chan_mid*cardin, chan_out, (1,1), init=df.init.prelu(), bias=False),
            mkbn(chan_out)
        ),
        identity_or_projection
    ], df.zoo.resnet.Add(), mknl())
