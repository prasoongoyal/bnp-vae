from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
from datetime import datetime
import sys
import os
import json
import numpy as np

from util import *
from layers import *
from io import open
from itertools import imap

DEFAULT_ARCH = [np.prod(IMG_DIM), 1024, 1024, NUM_PATHS]
DEFAULT_LEARNING_RATE = 1E-3

class Model(object):
  # TODO: Add arguments for architecture, learning rate, etc.
  def __init__(self, batch_size, output_dir):
    self.batch_size = batch_size
    self.architecture = DEFAULT_ARCH
    self.learning_rate = DEFAULT_LEARNING_RATE
    self.nonlinearity = tf.nn.elu
    self.squashing = tf.nn.sigmoid
    self.output_dir = output_dir

    # initialize path assignments (map of maps)
    self.path_assignments = {}

    self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    handles = self.buildGraph()
    self.session.run(tf.initialize_all_variables())
    
    (self.x_in, self.ncrp_prior, self.theta_normalized, self.x_reconstructed, 
        self.rec_cost_mean, self.kl_cost_mean, self.cost, self.global_step, 
        self.train_op, self.embedding) = handles

  def sampleMultinomial(self, theta_normalized):
    with tf.name_scope('sample_multinomial'):
      tmp = tf.reduce_mean(theta_normalized, axis=1, keep_dims=True, name='tmp')
      epsilon = tf.random_uniform(tf.shape(tmp), name='epsilon')
      theta_cumulative = tf.cumsum(theta_normalized, axis=1, 
                          name='theta_cumulative')
      excess_prob = tf.sub(theta_cumulative, epsilon, name='excess_prob')
      excess_prob_rounded = tf.sign(excess_prob, name='excess_prob_rounded')
      sample_idx = tf.argmax(excess_prob_rounded, 1, name='sample_idx')
      return sample_idx

  def buildGraph(self):
    #x_in = tf.placeholder(tf.float32, shape=[None, self.architecture[0]], name='x')
    x_in = tf.placeholder(tf.float32, shape=[None, IMG_DIM['width'], IMG_DIM['height'], 3], 
                          name='x')
    ncrp_prior = tf.placeholder(tf.float32, shape=[None, self.architecture[-1]], 
                                name='ncrp_prior')
    dropout = tf.placeholder_with_default(1.0, shape=[], name='dropout')

    CONV_FILTER_SIZES = [5, 5, 3]
    CONV_NUM_CHANNELS = [32, 32, 16]
    POOL_SIZES = [3, 4, 5]

    final_size = (int(IMG_DIM['width'] / np.prod(POOL_SIZES)), 
                  int(IMG_DIM['height'] / np.prod(POOL_SIZES)),
                  CONV_NUM_CHANNELS[-1])
    FC_SIZES = [int(np.prod(final_size)), 16]

    enc_conv_filters = {
      'layer1': tf.Variable(tf.random_normal([CONV_FILTER_SIZES[0], CONV_FILTER_SIZES[0], 
                                              IMG_DIM['channels'], CONV_NUM_CHANNELS[0]])),
      'layer2': tf.Variable(tf.random_normal([CONV_FILTER_SIZES[1], CONV_FILTER_SIZES[1], 
                                              CONV_NUM_CHANNELS[0], CONV_NUM_CHANNELS[1]])),
      'layer3': tf.Variable(tf.random_normal([CONV_FILTER_SIZES[2], CONV_FILTER_SIZES[2], 
                                              CONV_NUM_CHANNELS[1], CONV_NUM_CHANNELS[2]])),
    }
    enc_conv_biases = {
      'layer1': tf.Variable(tf.random_normal([CONV_NUM_CHANNELS[0]])),
      'layer2': tf.Variable(tf.random_normal([CONV_NUM_CHANNELS[1]])),
      'layer3': tf.Variable(tf.random_normal([CONV_NUM_CHANNELS[2]])),
    }

    dec_conv_filters = {
      'layer1': tf.Variable(tf.random_normal([CONV_FILTER_SIZES[2], CONV_FILTER_SIZES[2], 
                                              CONV_NUM_CHANNELS[2], CONV_NUM_CHANNELS[1]])),
      'layer2': tf.Variable(tf.random_normal([CONV_FILTER_SIZES[1], CONV_FILTER_SIZES[1], 
                                              CONV_NUM_CHANNELS[1], CONV_NUM_CHANNELS[0]])),
      'layer3': tf.Variable(tf.random_normal([CONV_FILTER_SIZES[0], CONV_FILTER_SIZES[0], 
                                              CONV_NUM_CHANNELS[0], IMG_DIM['channels']])),
    }
    dec_conv_biases = {
      'layer1': tf.Variable(tf.random_normal([CONV_NUM_CHANNELS[1]])),
      'layer2': tf.Variable(tf.random_normal([CONV_NUM_CHANNELS[0]])),
      'layer3': tf.Variable(tf.random_normal([IMG_DIM['channels']])),
    }

    fc_weights = {
      'layer1': tf.Variable(tf.random_normal([FC_SIZES[0], FC_SIZES[1]])),
      'layer2': tf.Variable(tf.random_normal([FC_SIZES[1], NUM_PATHS])),
      'layer3': tf.Variable(tf.random_normal([NUM_PATHS, FC_SIZES[1]])),
      'layer4': tf.Variable(tf.random_normal([FC_SIZES[1], FC_SIZES[0]])),
    }
    fc_biases = {
      'layer1': tf.Variable(tf.random_normal([FC_SIZES[1]])),
      'layer2': tf.Variable(tf.random_normal([NUM_PATHS])),
      'layer3': tf.Variable(tf.random_normal([FC_SIZES[1]])),
      'layer4': tf.Variable(tf.random_normal([FC_SIZES[0]])),
    }

    embedding =  tf.Variable(tf.random_normal(([NUM_PATHS, 10])))

    x_layer1 = Layers.conv_pool(x_in, enc_conv_filters['layer1'], 
                                enc_conv_biases['layer1'], CONV_FILTER_SIZES[0], POOL_SIZES[0])
    x_layer2 = Layers.conv_pool(x_layer1, enc_conv_filters['layer2'], 
                                enc_conv_biases['layer2'], CONV_FILTER_SIZES[1], POOL_SIZES[1])
    x_layer3 = Layers.conv_pool(x_layer2, enc_conv_filters['layer3'], 
                                enc_conv_biases['layer3'], CONV_FILTER_SIZES[2], POOL_SIZES[2])

    x_flatten = tf.reshape(x_layer3, [-1, int(np.prod(final_size))], name='x_flatten')

    x_fc1 = tf.add(tf.matmul(x_flatten, fc_weights['layer1']), fc_biases['layer1'])
    x_fc1_dropout = tf.nn.dropout(tf.tanh(x_fc1), dropout)
    x_fc2 = tf.add(tf.matmul(x_fc1_dropout, fc_weights['layer2']), fc_biases['layer2'])
    x_fc2_dropout = tf.nn.dropout(tf.tanh(x_fc2), dropout)

    theta_normalized = tf.nn.softmax(x_fc2_dropout)

    # reconstruction
    z_fc3 = tf.add(tf.matmul(theta_normalized, fc_weights['layer3']), fc_biases['layer3'])
    z_fc3_dropout = tf.nn.dropout(tf.tanh(z_fc3), dropout)
    z_fc4 = tf.add(tf.matmul(z_fc3_dropout, fc_weights['layer4']), fc_biases['layer4'])
    z_fc4_dropout = tf.nn.dropout(tf.tanh(z_fc4), dropout)

    z_reshape = tf.reshape(z_fc4_dropout, [-1, final_size[0], final_size[1], final_size[2]], 
                           name='z_reshape')
    z_layer1 =Layers.conv_unpool(z_reshape, dec_conv_filters['layer1'], 
                                dec_conv_biases['layer1'], CONV_FILTER_SIZES[2], POOL_SIZES[2])
    z_layer2 = Layers.conv_unpool(z_layer1, dec_conv_filters['layer2'], 
                                dec_conv_biases['layer2'], CONV_FILTER_SIZES[1], POOL_SIZES[1])
    x_reconstructed = tf.sigmoid(Layers.conv_unpool(z_layer2, dec_conv_filters['layer3'], 
                                dec_conv_biases['layer3'], CONV_FILTER_SIZES[0], POOL_SIZES[0]))
    
    rec_loss = Model.l2_loss(x_reconstructed, x_in)
    kl_loss = Model.kl_loss(theta_normalized, ncrp_prior)

    with tf.name_scope('rec_cost_mean'):
      rec_cost_mean = tf.reduce_mean(rec_loss)

    with tf.name_scope('kl_cost_mean'):
      kl_cost_mean = tf.reduce_mean(kl_loss)

    with tf.name_scope('cost'):
      cost = tf.reduce_mean(rec_loss + kl_loss, name='vae_cost')

    # optimization
    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope('Adam_optimizer'):
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      tvars = tf.trainable_variables()
      grads_and_vars = optimizer.compute_gradients(cost, tvars)
      train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step,
                  name='minimize_cost')

    return (x_in, ncrp_prior, theta_normalized, x_reconstructed, rec_cost_mean, 
            kl_cost_mean, cost, global_step, train_op, embedding)
    
  def train(self, train_data, max_iter=np.inf, max_epochs=np.inf, outdir='./out'):
    saver = tf.train.Saver(tf.all_variables())

    try:
      err_train = 0
      now = datetime.now().isoformat()[11:]

      print '------------ Training begin: {} -----------\n'.format(now)

      # initialize ncrp prior
      ncrp_prior = {}

      while True:
        x, x_annot, one_epoch_completed = train_data.get_next_batch()
        ncrp_prior_batch = []
        for (vidid, frameid) in x_annot:
          try:
            ncrp_prior_vid = ncrp_prior[vidid]
          except:
            ncrp_prior_vid = np.ones(NUM_PATHS)
            ncrp_prior_vid = ncrp_prior_vid / np.sum(ncrp_prior_vid)
          ncrp_prior_batch.append(ncrp_prior_vid)
        feed_dict = {self.x_in: x, self.ncrp_prior: ncrp_prior_batch}
        fetches = [self.x_reconstructed, self.theta_normalized, self.rec_cost_mean,
                  self.kl_cost_mean, self.cost, self.global_step, self.train_op, 
                  self.embedding]

        (x_reconstructed, theta_normalized, rec_cost_mean, kl_cost_mean, cost,
         iteration, _, embedding) = self.session.run(fetches, feed_dict)

        z = np.argmax(theta_normalized, 1)

        if iteration%1000 == 0:
          # write model
          saver.save(self.session, os.path.join(self.output_dir, 'model'), 
                     global_step = iteration)
          Model.write_z(self.path_assignments, os.path.join(self.output_dir,
                        'assignments'+unicode(iteration)+'.txt'))

        for i in xrange(len(z)):
          (vidid, frameid) = x_annot[i]
          try: 
            vid_path_assignments = self.path_assignments[vidid]
          except:
            vid_path_assignments = {}
          
          vid_path_assignments[frameid] = int(z[i])
          self.path_assignments[vidid] = vid_path_assignments

        if iteration%1 == 0:
          ncrp_prior = Model.recompute_ncrp(self.path_assignments)

        err_train += cost
        print (('Iter : %d \t ' +
                'Rec. loss : %f \t' +
                'KL-div loss : %f \t' +
                'Total loss : %f') % (iteration, rec_cost_mean, kl_cost_mean, cost))

    except KeyboardInterrupt:
        now = datetime.now().isoformat()[11:]
        print '---------- Training end: {} -----------\n'.format(now)
        sys.exit(0)

  @staticmethod
  def normalize_prob_crp(prob):
    # replace zeros with GAMMA
    zero_indices = [i for (i, f) in enumerate(prob) if f==0]
    try:
      default_prob = GAMMA / len(zero_indices)
      prob = list(imap(lambda x:default_prob if x==0 else x, prob))
    except:
      pass
    #normalize
    s = float(sum(prob))
    if s==0:
      return prob
    else:
      prob_norm = list(imap(lambda x: x/s, prob))
      return prob_norm

  @staticmethod
  def compute_ncrp_prior(path_freq):
    if len(path_freq) == BRANCHING_FACTOR:
      return Model.normalize_prob_crp(path_freq)
    else:
      # create higher level frequencies
      parent_freq = []
      for i in xrange(int(len(path_freq)/BRANCHING_FACTOR)):
        parent_freq.append(sum(path_freq[i*BRANCHING_FACTOR:(i+1)*BRANCHING_FACTOR]))
      # compute probabilities for parents recursively
      parent_prob = Model.compute_ncrp_prior(parent_freq)
        
      # compute probabilities for current level
      prob = []
      for i in xrange(len(parent_freq)):
        prob += imap(lambda x: parent_prob[i] * x,
                Model.normalize_prob_crp(path_freq[i*BRANCHING_FACTOR:
                                          (i+1)*BRANCHING_FACTOR]))
      return prob  

  @staticmethod
  def recompute_ncrp(path_assignments):
    ncrp_priors = {}
    for vidid in path_assignments:
      path_freq = [0] * NUM_PATHS
      for frameid in path_assignments[vidid]:
        path_freq[int(path_assignments[vidid][frameid])] += 1
      ncrp_priors[vidid] = Model.compute_ncrp_prior(path_freq)
    return ncrp_priors
        
  @staticmethod
  def write_z(z, filename):
    fileptr = open(filename, 'w')
    for vidid in xrange(500):
      try:
        vid_assignments = z[unicode(vidid)]
      except:
        vid_assignments = {}
      if len(vid_assignments) > 0:
        fileptr.write(unicode(vidid) + '\t' + 
                      ' '.join(imap(unicode, vid_assignments.values())) + '\n')
    fileptr.close()

  @staticmethod
  def kl_loss(theta_normalized, ncrp_prior):
    offset = 1e-7
    with tf.name_scope('kl_loss'):
      theta_normalized_ = tf.clip_by_value(theta_normalized, offset, 1-offset)
      ncrp_prior_ = tf.clip_by_value(ncrp_prior, offset, 1-offset)
      log_theta_normalized = tf.log(theta_normalized_, name='log_theta_normalized')
      log_ncrp_prior = tf.log(ncrp_prior_, name='log_ncrp_prior')
      log_diff = tf.subtract(log_theta_normalized, log_ncrp_prior, name='log_diff')
      return tf.reduce_sum(tf.multiply(theta_normalized_, log_diff), 1)
    

  @staticmethod
  def l2_loss(obs, actual):
    with tf.name_scope('l2_loss'):
      return tf.reduce_mean(tf.square(obs - actual), [1, 2, 3])
