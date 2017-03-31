import DeepFried2 as df


def expand_dims(a, axis):
    idx = [slice(None)]*(a.ndim+1)
    idx[axis] = None
    return a[tuple(idx)]


def cdist_theano(x, eps=1e-8, squared=False):
    # Extend x as x[:,None,...] (but ... doesn't exist in Theano yet.)
    # Then, thanks to broadcasting, we get (B,1,...) - (B,...) = (B,B,...)
    diff = expand_dims(x, 1) - x
    # Finally, sum over all axes but the two first ones.
    #return df.T.sum(diff*diff, axis=tuple(range(2, diff.ndim)))
    dsq = df.T.sum(diff*diff, axis=tuple(range(2, diff.ndim)))
    if squared:
        return dsq
    else:
        return df.T.sqrt(eps + dsq)


class OutputNormPenalty:
    def __init__(self, module, value=1, axis=-1, eps=1e-8):
        self._mod = module
        self._v = value
        self._axis = axis
        self._eps = eps

    def symb_forward(self):
        x = self._mod._last_symb_out[self._mod._mode]
        norms = df.T.sqrt(self._eps + df.T.sqr(x).sum(axis=self._axis))
        return df.T.mean(abs(norms - self._v))


class OutputDistancePenalty:
    """
    Type can be either 'exp' or 'inv'.

    - 'inv': alpha is the number at which the penalty is 1.
             Lower distance than that dramatically increases penalty,
             higher distance quickly vanishes.
    - 'exp': penalty never over 1.0, more S-like curve.
             `alpha` is where the curve hits turning-point, at ~0.5 penalty.
             About 1 order of mag smaller it hits ~0.95, and about 4x larger
             it hits ~0.05.
    """
    def __init__(self, module, type='exp', alpha=1):
        self._mod = module
        self._alpha = alpha
        self._typ = type

    def symb_forward(self):
        symb_x = self._mod._last_symb_out[self._mod._mode]
        dists = cdist_theano(symb_x)

        if self._typ == 'exp':
            return df.T.mean(df.T.exp(-(1/self._alpha)*dists))
        elif self._typ == 'inv':
            return df.T.mean(self._alpha/dists)


class BatchHardCriterion(df.Criterion):
    def __init__(self, margin=None, donorm=True, normsq=False, normeps=1e-8):
        df.Criterion.__init__(self)
        self.margin = margin
        self.donorm = donorm
        self.normsq = normsq
        self.normeps = normeps

    def symb_forward(self, symb_embs, symb_pids):
        # TODO: Multiple embeddings for the same person? Trains crops for free/hi-res maps?

        # Flatten all features, so we got (B,F) and everything is easier.
        symb_embs = df.T.flatten(symb_embs, 2)

        dists = cdist_theano(symb_embs)

        # Mask of (B,B) with positive/negative pairs.
        poss = df.T.cast(df.T.eq(symb_pids[:,None], symb_pids), df.floatX)
        negs = df.T.cast(df.T.neq(symb_pids[:,None], symb_pids), df.floatX)

        # Find the worst offenders.
        furthest_pos, _ = df.th.scan(lambda ds, mask: df.T.max(ds[mask.nonzero()]), sequences=[dists, poss])
        closest_neg, _ = df.th.scan(lambda ds, mask: df.T.min(ds[mask.nonzero()]), sequences=[dists, negs])

        diff = furthest_pos - closest_neg
        if self.margin is not None:
            diff = df.T.maximum(diff + self.margin, 0.0)
        else:
            diff = df.T.nnet.nnet.softplus(diff)

        if not self.donorm:
            return diff

        # And the extra norm term too!
        norms = df.T.sqrt(self.normeps + df.T.sum(symb_embs*symb_embs, axis=1))
        d = df.T.log(norms)
        if self.normsq:
            logdist_from_wanted = d*d
        else:
            logdist_from_wanted = abs(d)

        return df.T.concatenate([diff, logdist_from_wanted])


class BatchAllCriterion(df.Criterion):
    def __init__(self, margin=None, avg_nonzeros=False, topk=False):
        df.Criterion.__init__(self)
        if avg_nonzeros:
            self.enable_nonzero_averaging()

        self.margin = margin
        self.topk = topk

    def symb_forward(self, symb_x, symb_pids):
        # Flatten all features, so we got (B,F) and everything is easier.
        symb_x = df.T.flatten(symb_x, 2)

        dists = cdist_theano(symb_x)

        # Mask of (B,B) with positive/negative pairs.
        poss = df.T.cast(df.T.eq(symb_pids[:,None], symb_pids), dtype='int32')
        pos2 = poss - df.T.eye(poss.shape[0], dtype='int32')
        negs = 1 - poss

        def per_sample(dists_, poss_, negs_):
            # dists_ is a 1D array of distances
            # poss_ is a 1D array of 1.0 for positives and 0.0 else
            # negs_ is a 1D array of 1.0 for negatives and 0.0 else
            # notice how "self" is not 1.0 in either!
            pos_dists = dists_[poss_.nonzero()]
            neg_dists = dists_[negs_.nonzero()]

            # Now compute a (P,N)-matrix of all-to-all (pos - neg) differences.
            all_diff = pos_dists[:,None] - neg_dists[None,:]
            if self.margin is not None:
                all_diff = df.T.maximum(all_diff + self.margin, 0.0)
            else:
                all_diff = df.T.nnet.nnet.softplus(all_diff)

            if self.topk is not False:
                # NOTE: This part only works for fixed-size batches.
                #all_diff = df.T.sort(all_diff, axis=None)[-self.topk:]
                return all_diff

            if self._nonzero_averaging:
                nnz = df.th.gradient.disconnected_grad(all_diff.nonzero_values().shape[0])
                return df.T.sum(all_diff)/(1e-8 + nnz)
            else:
                return df.T.mean(all_diff)

        # Just loop over each sample, because each sample may have a different
        # number of positives/negatives, we can't do this in a tensor-expr.
        per_sample_loss, _ = df.th.scan(per_sample, sequences=[dists, pos2, negs])

        if self.topk is not False:
            per_sample_loss = df.T.sort(per_sample_loss, axis=None)[-self.topk:]

        return per_sample_loss


class NormCriterion(df.Criterion):
    def __init__(self, wanted_norm=1, eps=1e-8, square=False):
        df.Criterion.__init__(self)
        self.eps = eps
        self.wanted_norm = wanted_norm
        self.square = square

    def symb_forward(self, symb_embs, symb_ious):
        # Flatten all features, so we got (B,F) and everything is easier.
        symb_embs = df.T.flatten(symb_embs, 2)

        norms = df.T.sqrt(self.eps + df.T.sum(symb_embs*symb_embs, axis=1))

        d = df.T.log(norms/self.wanted_norm)
        if self.square:
            logdist_from_wanted = d*d
        else:
            logdist_from_wanted = abs(d)

        # Given the eps above, here norms is guaranteed not to be exact 0!
        return df.T.power((1 - symb_ious) + logdist_from_wanted, 2*symb_ious-1)


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
