from __future__ import absolute_import
import numpy as np
import tensorflow as tf
import sys
import functools
from functional import partial, compose

batch_size = 2
IMG_DIM = {'width': 480, 'height': 360, 'channels': 3}
#GAMMA = 1.0   # for ncrp
BRANCHING_FACTOR = 3
NUM_LEVELS = 4
NUM_INTERNAL_EDGES = BRANCHING_FACTOR * ((BRANCHING_FACTOR ** (NUM_LEVELS-2) - 1) /
                                         (BRANCHING_FACTOR - 1))
NUM_PATHS = BRANCHING_FACTOR ** (NUM_LEVELS - 1)
NUM_EDGES = BRANCHING_FACTOR * ((BRANCHING_FACTOR ** (NUM_LEVELS-1) - 1) / 
                                (BRANCHING_FACTOR - 1))

def composeAll(*args):
    """Util for multiple function composition

    i.e. composed = composeAll([f, g, h])
         composed(x) # == f(g(h(x)))
    """
    # adapted from https://docs.python.org/3.1/howto/functional.html
    return partial(functools.reduce, compose)(*args)

