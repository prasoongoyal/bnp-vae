import numpy as np
import tensorflow as tf
import functools
import sys
from functional import compose, partial

IMG_DIM = (640, 360, 3)
GAMMA = 1.0
BRANCHING_FACTOR = 3
NUM_LEVELS = 4
NUM_PATHS = BRANCHING_FACTOR ** (NUM_LEVELS - 1)

def get_videoid_frameid(path):
  try:
    # remove extension
    path = path[:-4]
    filename = path[path.rfind('/')+1:]
    #print filename, path
    videoid, frameid = filename.split('_')
    videoid = eval(videoid[3:])
    frameid = eval(frameid[1:])
    return videoid, frameid
  except:
    sys.exit('Invalid file name format!')

def composeAll(*args):
  """Util for multiple function composition

  i.e. composed = composeAll([f, g, h])
       composed(x) # == f(g(h(x)))
  """
  # adapted from https://docs.python.org/3.1/howto/functional.html
  return partial(functools.reduce, compose)(*args)

