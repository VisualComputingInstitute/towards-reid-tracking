# Towards a Principled Integration of Multi-Camera Re-Identification and Tracking through Optimal Bayes Filters

This is the code for reproducing the experiments from our paper [Towards a Principled Integration of Multi-Camera Re-Identification and Tracking through Optimal Bayes Filters](https://arxiv.org/abs/1705.04608).

Please note that this is very much research code.
It's made public for reference so that others can see what exactly we did, as the paper in no way can explain everything in enough detail.
**It is not production-quality code**, rather it is nice code that got ever more messy as the deadline approached.

### The neural networks

The training code of the neural networks is not public yet as it's pending publication of the dependency at https://github.com/VisualComputingInstitute/triplet-reid.

However, the code creating the models and loading the trained weights is included.
It is based on a custom deep-learning library on top of Theano called [DeepFried2](https://github.com/lucasb-eyer/DeepFried2) and a small toolbox called [lbtoolbox](https://github.com/lucasb-eyer/lbtoolbox) that you'll need to install.
This can be easily done using `pip install -e git+GITHUB_URL`, see the corresponding READMEs.

The model we used for final experiments is `lunet2c` and [the weights we used can be downloaded here](https://omnomnom.vision.rwth-aachen.de/data/lunet2c-noscale-nobg-2to32-aug.pkl).


