from __future__ import absolute_import
import os
import sys

import numpy as np
import tensorflow as tf

from data import Data
from model import Model
from model_conv import Model

#import vae

from numpy.random import permutation

def main(files_list, output_dir, resume_training = None):
  #data = load_data(files_list)
  train_data = Data(files_list, 32)
  model = Model(batch_size=32, output_dir=output_dir)
  model.train(train_data)

if __name__ == u"__main__":
  tf.reset_default_graph()

  files_list = sys.argv[1]
  output_dir = sys.argv[2]
  main(files_list, output_dir)
  '''
  try:
    resume_training = sys.argv[2]
    main(files_list, resume_training)
  except IndexError:
    main(files_list)
  '''
