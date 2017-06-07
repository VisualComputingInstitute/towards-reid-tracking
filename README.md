## Towards a Principled Integration of Multi-Camera Re-Identification and Tracking through Optimal Bayes Filters

This is the code for reproducing the experiments from our paper [Towards a Principled Integration of Multi-Camera Re-Identification and Tracking through Optimal Bayes Filters](https://arxiv.org/abs/1705.04608).
If you end up using any of this in your publication or otherwise find it useful, please cite our work as:

```
@article{BeyerBreuers2017Arxiv,
  author    = {Lucas Beyer and
               Stefan Breuers and
               Vitaly Kurin and
               Bastian Leibe},
  title     = {Towards a Principled Integration of Multi-Camera Re-Identification
               and Tracking through Optimal Bayes Filters},
  journal   = {arXiv preprint arXiv:1705.04608},
  year      = {2017},
}
```

Please note that this is very much research code, and the paper is a very exploratory one.
It's made public for reference so that others can see what exactly we did, as the paper in no way can explain everything in enough detail.
**It is not production-quality code**, rather it is nice code that got ever more messy as the deadline approached.

Due to the nature of the code, many things might still be confusing and non-obvious to others, so feel free to ask us, either by opening an issue here on github (preferably), or shooting us an e-mail!

## The neural networks

The training code of the neural networks is not public yet as it's pending publication of the dependency at https://github.com/VisualComputingInstitute/triplet-reid.

However, the code creating the models and loading the trained weights is included.
It is based on a custom deep-learning library on top of Theano called [DeepFried2](https://github.com/lucasb-eyer/DeepFried2) and a small toolbox called [lbtoolbox](https://github.com/lucasb-eyer/lbtoolbox) that you'll need to install.
This can be easily done using `pip install -e git+GITHUB_URL`, see the corresponding READMEs.

The model we used for final experiments is `lunet2c` and [the weights we used can be downloaded here](https://omnomnom.vision.rwth-aachen.de/data/lunet2c-noscale-nobg-2to32-aug.pkl).

## The run parameters

```
NN-KF
DIST_THRESH = 200, det_init_thresh = 0.3, det_continue_thresh = 0.0 init_thresh = 3, delete_thresh = 5

+GT init
--gt_init
DIST_THRESH = 200, DET_INIT_THRESH = 0.3, DET_CONTINUE_THRESH = -0.3, init_thresh=1, delete_thresh=90

+ReID
--gt_init --use_appearance
DIST_THRESH = 200, APP_THRESH = 6, DET_INIT_THRESH = 0.3, DET_CONTINUE_THRESH = -0.3, init_thresh=1, delete_thresh=90

only ReID
--gt_init --use_appearance
DIST_THRESH = 6, DET_INIT_THRESH = 0.3, DET_CONTINUE_THRESH = -0.3, init_thresh=1, delete_thresh=90

Full
--dist_thresh 6 --unmiss_thresh 2

+entropy
--dist_thresh 5.5 --ent_thresh 0.25 --maxlife 8000 --unmiss_thresh 5
killed of age: 4
```

Final raw bounding box results [can be found here](https://omnomnom.vision.rwth-aachen.de/data/bbmtrack-results/).

