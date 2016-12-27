from __future__ import absolute_import
import numpy as np
import tensorflow as tf
import sys

IMG_DIM = (480, 360, 3)
GAMMA = 1.0
BRANCHING_FACTOR = 3
NUM_LEVELS = 4
NUM_PATHS = BRANCHING_FACTOR ** (NUM_LEVELS - 1)

def get_videoid_frameid(path):
  try:
    # remove extension
    path = path[:-4]
    filename = path[path.rfind(u'/')+1:]
    #print filename, path
    videoid, frameid = filename.split(u'_')
    videoid = eval(videoid[3:])
    frameid = eval(frameid[1:])
    return videoid, frameid
  except:
    sys.exit(u'Invalid file name format!')

