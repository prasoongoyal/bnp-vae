from __future__ import absolute_import
import os
import sys

#from data_shards import Data
from data import Data
from model import Model
from util import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--files_list', type=str, help='Text file containing a list of paths of all trianing images')
parser.add_argument('--output_dir', type=str, help='Output directory')
parser.add_argument('--kl_coeff', type=float, default=1.0, help='Coefficient of KL-div term in the loss function.')
parser.add_argument('--files_list', type=str, help='Model file (required for predict.py only)')
parser.add_argument('--predict_output_file', type=str, help='File where predictions are to be written (required for predict.py only)')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and prediction.')
parser.add_argument('--latent_code_sizze', type=int, default=32, help='Latent code size.')
parser.add_argument('--stdev_thr', type=int, default=32, help='Threshold for standard deviation for dynamically modifying the tree.')
parser.add_argument('--num_splits', type=int, default=4, help='No. of nodes to split a node into for dynamically modifying the tree.')

args = parser.parse_args()

#TODO: Add an option to resume training from pretrained model.
def main(files_list, output_dir):
  train_data = Data(files_list, '/work/ans556/prasoon/Features/full_data_valid_th.npy')
  model = Model(batch_size=batch_size, output_dir=output_dir)
  #model.train(train_data)
  model.predict(train_data)
  #model.log_likelihood(train_data)

if __name__ == u"__main__":
  tf.reset_default_graph()

  # set random seeds
  #tf.set_random_seed(17);
  #np.random.seed(13)

  files_list = args.files_list
  output_dir = args.output_dir

  main(files_list, output_dir)
