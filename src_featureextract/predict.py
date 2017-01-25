from __future__ import absolute_import
import os
import sys

#from data_shards import Data
from data import Data
from model import Model
from util import *


#TODO: Add an option to resume training from pretrained model.
def main(files_list, output_dir):
  train_data = Data(files_list, '/scratch/ans556/prasoon/data/MED/med_train_5_th.npy')
  model = Model(batch_size=batch_size, output_dir=output_dir)
  #model.train(train_data)
  model.predict(train_data)

if __name__ == u"__main__":
  tf.reset_default_graph()

  # set random seeds
  tf.set_random_seed(17);
  np.random.seed(13)

  files_list = sys.argv[1]
  output_dir = sys.argv[2]

  main(files_list, output_dir)
