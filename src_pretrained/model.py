from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
from datetime import datetime
import sys
import os
import json
import numpy as np

from copy import deepcopy
from scipy.special import digamma
from util import *
from layers import *
from var_inf import VarInf
from io import open
from itertools import imap

#DEFAULT_ARCH = [np.prod(IMG_DIM), 1024, 1024, NUM_PATHS]
DEFAULT_LEARNING_RATE = 1E-4

class Model(object):
  # TODO: Add arguments for architecture, learning rate, etc.
  def __init__(self, batch_size, output_dir):
    self.batch_size = batch_size
    self.learning_rate = DEFAULT_LEARNING_RATE
    self.nonlinearity = tf.nn.elu
    self.squashing = tf.nn.sigmoid
    self.output_dir = output_dir

    # initialize path assignments (map of maps)
    self.path_assignments = {}

    self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    handles = self.buildGraph()
    self.session.run(tf.initialize_all_variables())
    
    (self.x_in, self.mu_in, self.log_sigma_in, self.z, self.z_mean, 
     self.z_log_sigma, self.x_reconstructed, 
     self.rec_cost_mean, self.kl_cost_mean, self.cost, self.global_step, 
     self.train_op) = handles

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
    # network parameters
    CONV_FILTER_SIZES = [5, 5, 3]
    CONV_NUM_CHANNELS = [IMG_DIM['channels'], 32, 32, 16]
    POOL_SIZES = [3, 4, 5]

    final_size = (int(IMG_DIM['width'] / np.prod(POOL_SIZES)), 
                  int(IMG_DIM['height'] / np.prod(POOL_SIZES)),
                  CONV_NUM_CHANNELS[-1])
    FC_SIZES = [int(np.prod(final_size)), 16, 16]

    # network inputs
    x_in = tf.placeholder(tf.float32, shape=[None, IMG_DIM['width'], IMG_DIM['height'], 3], 
                          name='x')
    mu_in = tf.placeholder(tf.float32, shape=[None, LATENT_CODE_SIZE])
    log_sigma_in = tf.placeholder(tf.float32, shape=[None, 1])
    dropout = tf.placeholder_with_default(1.0, shape=[], name='dropout')

    enc_conv_layers = [ConvPool('encoder_conv_pool', conv_kernel_size, conv_output_channels,
                      pool_size, tf.tanh) for (conv_kernel_size, conv_output_channels, 
                      pool_size) in zip(CONV_FILTER_SIZES, CONV_NUM_CHANNELS[1:], POOL_SIZES)]
    x_conv = composeAll(reversed(enc_conv_layers))(x_in)

    x_flatten = tf.reshape(x_conv, [-1, int(np.prod(final_size))], name='x_flatten')

    enc_fc_layers = [Dense('enc_fc', output_size, dropout, tf.tanh) 
                     for output_size in FC_SIZES[1:]]
    x_encoded = composeAll(reversed(enc_fc_layers))(x_flatten)

    # mean and standard deviation for sampling latent code
    z_mean= Dense('z_mean', LATENT_CODE_SIZE)(x_encoded)
    z_log_sigma = Dense('z_log_std', LATENT_CODE_SIZE)(x_encoded)

    # sample latent code
    z = self.sampleGaussian(z_mean, z_log_sigma)

    # reconstruction
    dec_fc_layers = [Dense('dec_fc', output_size, dropout, tf.tanh)
                     for output_size in FC_SIZES[:-1]]
    z_fc = composeAll(dec_fc_layers)(z)

    z_reshape = tf.reshape(z_fc, [-1, final_size[0], final_size[1], final_size[2]], 
                           name='z_reshape')

    dec_conv_layers = [DeconvUnpool('decoder_deconv_unpool', deconv_kernel_size, 
                      deconv_output_channels, unpool_size, tf.tanh) for (deconv_kernel_size,
                      deconv_output_channels, unpool_size) in zip(CONV_FILTER_SIZES,
                      CONV_NUM_CHANNELS[:-1], POOL_SIZES)]
    x_reconstructed = 2.0/np.pi * tf.tanh(composeAll(dec_conv_layers)(z_reshape))

    rec_loss = Model.l2_loss(x_reconstructed, x_in)
    kl_loss = Model.kl_loss(z_mean, z_log_sigma, mu_in, log_sigma_in)

    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('rec_cost_mean'):
      rec_cost_mean = tf.reduce_mean(rec_loss)

    with tf.name_scope('kl_cost_mean'):
      kl_cost_mean = tf.reduce_mean(kl_loss)

    with tf.name_scope('cost'):
      cost = tf.reduce_mean(rec_loss + kl_loss, 
                            name='vae_cost')

    # optimization
    with tf.name_scope('Adam_optimizer'):
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      tvars = tf.trainable_variables()
      grads_and_vars = optimizer.compute_gradients(cost, tvars)
      train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step,
                  name='minimize_cost')

    return (x_in, mu_in, log_sigma_in, z, z_mean, z_log_sigma, x_reconstructed, 
            rec_cost_mean, kl_cost_mean, cost, global_step, train_op)
    
  def train(self, train_data, max_iter=np.inf, max_epochs=np.inf, outdir='./out'):
    saver = tf.train.Saver(tf.all_variables())

    self.latent_codes = {}

    var_inf = VarInf()

    try:
      err_train = 0
      training_start_time = datetime.now()
      now = datetime.now().isoformat()[11:]

      print '------------ Training begin: {} -----------\n'.format(now)

      # initialize ncrp prior
      ncrp_prior = {}

      while True:
        x, x_annot, one_epoch_completed = train_data.get_next_batch()
        mu_true_path = var_inf.get_true_path_mean(x_annot)
        batch_size = np.size(x, 0)
        log_sigma_true_path = np.log(SIGMA_Z) * np.ones(shape=(batch_size, 1))
        feed_dict = {self.x_in: x, 
                     self.mu_in: mu_true_path,
                     self.log_sigma_in: log_sigma_true_path}
        fetches = [self.z, self.z_mean, self.z_log_sigma, 
                   self.x_reconstructed, self.rec_cost_mean,
                   self.kl_cost_mean, self.cost, self.global_step, self.train_op]

        (z, z_mean, z_log_sigma, x_reconstructed, rec_cost_mean, kl_cost_mean, cost,
         iteration, _) = self.session.run(fetches, feed_dict)

        self.update_latent_codes(z, x_annot)

        # update variational parameters periodically and save current state
        if iteration%1000 == 0:
          saver.save(self.session, os.path.join(self.output_dir, 'model'), 
                     global_step = iteration)
          self.write_latent_codes(os.path.join(self.output_dir, 
                                               'z_'+unicode(iteration)+'.txt'))
          var_inf.update_variational_parameters(self.latent_codes)
          var_inf.write_alpha(os.path.join(self.output_dir, 
                                           'alpha_'+unicode(iteration)+'.txt'))

        err_train += cost
        print (('Iter : %d \t ' +
                'Rec. loss : %f \t' +
                'KL-div loss : %f \t' +
                'Total loss : %f') % (iteration, rec_cost_mean, kl_cost_mean, cost))

    except KeyboardInterrupt:
        now = datetime.now().isoformat()[11:]
        print '---------- Training end: {} -----------\n'.format(now)
        # write model
        saver.save(self.session, os.path.join(self.output_dir, 'model'), 
                   global_step = iteration)
        sys.exit(0)

  def update_latent_codes(self, z_batch, x_annot_batch):
    for (z, x_annot) in zip(z_batch, x_annot_batch):
      (vidid, frameid) = x_annot
      try:
        latent_codes_vid = self.latent_codes[vidid]
      except KeyError:
        latent_codes_vid = {}
      latent_codes_vid[frameid] = z
      self.latent_codes[vidid] = latent_codes_vid

  def sampleGaussian(self, mu, log_sigma):
    # (Differentiably!) draw sample from Gaussian with given shape, 
    # subject to random noise epsilon
    with tf.name_scope("sample_gaussian"):
      # reparameterization trick
      epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
      return mu + epsilon * tf.exp(log_sigma) # N(mu, I * sigma**2)

  def write_latent_codes(self, filename):
    with open(filename, 'w') as f:
      for vidid in self.latent_codes:
        for frameid in self.latent_codes[vidid]:
          f.write(vidid + '\t' + frameid + '\t' + ' '.join(map(str, \
                         self.latent_codes[vidid][frameid])) + '\n')

  @staticmethod
  def kl_loss(mu_pred, log_sigma_pred, mu_in, log_sigma_in):
    # offset = 1e-7
    with tf.name_scope('kl_loss'):
      kl_loss = tf.reduce_sum(tf.subtract(log_sigma_in, log_sigma_pred), axis=1)
      mu_diff_scaled = tf.div(tf.subtract(mu_in, mu_pred), tf.exp(log_sigma_in))
      kl_loss = tf.add(kl_loss, tf.reduce_sum(tf.square(mu_diff_scaled), axis=1))
      kl_loss = tf.add(kl_loss, tf.reduce_sum(tf.div(tf.exp(2.0 * log_sigma_pred), 
                                       tf.exp(2.0 * log_sigma_in)), axis=1))
      return kl_loss

  @staticmethod
  def l2_loss(obs, actual):
    with tf.name_scope('l2_loss'):
      return tf.reduce_mean(tf.square(obs - actual), [1, 2, 3])
