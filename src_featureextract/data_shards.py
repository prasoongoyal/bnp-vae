from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
from PIL import Image
from io import open
import numpy as np
from util import *

class Data(object):
  def __init__(self, files_list, data_file):
    self.files_list = files_list
    self.batch_size = batch_size
    #self.data = np.load(data_file)
    self.curr_shard_idx = 1
    self.curr_shard = np.load('/scratch/ans556/prasoon/MED11/features_shards/med_features_' + 
                              str(self.curr_shard_idx) + '_th.npy')
    self.num_shards = 9
    self.shards_cum_length = [0, 49999, 99999, 149999, 199999, 249999, 299999, 349999, 399999,
                              427850]
    self.metadata = self.prepare_data()
    #print self.metadata
    self.one_epoch_completed = False
    self.batch_start_idx = 0

  def prepare_data(self):
    train_data = []
    idx = 0
    with open(self.files_list) as f:
      for line in f.readlines():
        path = line.strip()
        videoid, frameid = Data.get_videoid_frameid(path)
        train_data.append((idx, videoid, frameid))
        idx += 1
    #print idx, np.shape(self.data)
    #print train_data[:10]
    # shuffle data
    #train_data = np.random.permutation(train_data)
    #np.random.shuffle(train_data)
    for i in range(1, self.num_shards+1):
      data_i = train_data[self.shards_cum_length[i-1]:self.shards_cum_length[i]]
      np.random.shuffle(data_i)
      train_data[self.shards_cum_length[i-1]:self.shards_cum_length[i]] = data_i
    #print train_data[:10]
    print u'Training on %d image files...' % len(train_data)
    return train_data

  def get_next_batch(self):
    curr_batch = self.metadata[self.batch_start_idx:
                  min(self.batch_start_idx+self.batch_size, 
                      self.shards_cum_length[self.curr_shard_idx])]
    batch = []
    batch_annot = []
    data_indices = map(lambda x: x[0] - self.shards_cum_length[self.curr_shard_idx-1], 
                       curr_batch)
    #print curr_batch
    #print data_indices
    batch = self.curr_shard[data_indices, :]
    batch_annot = map(lambda x: (x[1], x[2]), curr_batch)
    self.batch_start_idx += self.batch_size
    if (self.batch_start_idx >= self.shards_cum_length[self.curr_shard_idx]):
      # load next shard
      self.curr_shard_idx += 1
      self.batch_start_idx = self.shards_cum_length[self.curr_shard_idx-1]
      if (self.curr_shard_idx > self.num_shards):
        self.curr_shard_idx = 1
        self.batch_start_idx = 0
        self.one_epoch_completed = True
      self.curr_shard = np.load('/scratch/ans556/prasoon/MED11/features_shards/med_features_' + 
                                str(self.curr_shard_idx) + '_th.npy')
      print 'Loaded next shard'
      
    '''
    if (self.batch_start_idx >= len(self.metadata)):
      self.batch_start_idx = 0
      self.one_epoch_completed = True
    '''
      #self.metadata = np.random.permutation(self.data)
      #np.random.shuffle(self.metadata)
    # load images and preprocess
    return batch, batch_annot, self.one_epoch_completed

  # gets image filename formatted as "path/to/dir/vid<vidid>_f<frameid>.jpg"
  @staticmethod
  def get_videoid_frameid(path):
    try:
      path = path[:-4]                                      # remove extension
      filename = path[path.rfind(u'/')+1:]                  # remove path to directory
      videoid, frameid = filename.split(u'_')               # split videoid and frameid
      videoid = videoid[3:]                                 # extract videoid
      frameid = frameid[1:]                                 # extract frameid
      return videoid, frameid
    except:
      sys.exit(u'Invalid file name format!')

    