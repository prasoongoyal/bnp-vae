import os
import sys

import numpy as np
import tensorflow as tf

from data import Data
from model import Model

#import vae

from numpy.random import permutation

def main(files_list, resume_training = None):
  #data = load_data(files_list)
  train_data = Data(files_list, 128)
  model = Model(batch_size=128)
  model.train(train_data)

if __name__ == "__main__":
  tf.reset_default_graph()

  files_list = sys.argv[1]
  try:
    resume_training = sys.argv[2]
    main(files_list, resume_training)
  except IndexError:
    main(files_list)
