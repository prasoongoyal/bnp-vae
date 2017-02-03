from __future__ import absolute_import
import os
import sys

#from data_shards import Data
from data import Data
from model import Model
from util import *
#import util
import tensorflow as tf
import numpy as np

#TODO: Add an option to resume training from pretrained model.
def main(files_list, output_dir):
  train_data = Data(files_list, '/work/ans556/prasoon/Features/full_data_th.npy')
  model = Model(batch_size=batch_size, output_dir=output_dir)
  model.train(train_data)

if __name__ == u"__main__":
  tf.reset_default_graph()

  # set random seeds
  tf.set_random_seed(17);
  np.random.seed(13)

  files_list = sys.argv[1]
  output_dir = sys.argv[2]

  #util.BRANCHING_FACTOR = eval(sys.argv[3])
  #util.NUM_LEVELS = eval(sys.argv[4])

  main(files_list, output_dir)
