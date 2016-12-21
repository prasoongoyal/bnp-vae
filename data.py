import numpy as np
import tensorflow as tf

from numpy.random import permutation

from util import *
from PIL import Image

class Data():
  def __init__(self, files_list, batch_size):
    self.files_list = files_list
    self.batch_size = batch_size
    self.data = self.prepare_data()
    self.one_epoch_completed = False

    self.batch_start_idx = 0

  def prepare_data(self):
    train_data = []
    with open(self.files_list) as f:
      for line in f.readlines():
        path = line.strip()
        videoid, frameid = get_videoid_frameid(path)
        train_data.append((path, videoid, frameid))
    # shuffle data
    train_data = permutation(train_data)
    print ('Training on %d image files...' % len(train_data))
    return train_data
  
  def get_next_batch(self):
    curr_batch = self.data[self.batch_start_idx:
                  self.batch_start_idx+self.batch_size] #works even for last batch
    self.batch_start_idx += self.batch_size
    if (self.batch_start_idx > len(self.data)):
      self.batch_start_idx = 0
      self.one_epoch_completed = True
    # load images
    batch = []
    batch_annot = []
    for image_info in curr_batch:
      img = Image.open(image_info[0])
      img = img.resize((36, 64))
      img_arr = np.asarray(img)
      img_arr = img_arr / 255.0
      #print (np.shape(img_arr))
      batch.append(np.ndarray.flatten(img_arr))
      batch_annot.append((image_info[1], image_info[2]))

    #print (len(result))
    return batch, batch_annot, self.one_epoch_completed
