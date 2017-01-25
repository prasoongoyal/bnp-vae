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
DEFAULT_LEARNING_RATE = 1E-3

class Model(object):
  # TODO: Add arguments for architecture, learning rate, etc.
  def __init__(self, batch_size, output_dir):
    self.batch_size = batch_size
    self.learning_rate = DEFAULT_LEARNING_RATE
    self.squashing = tf.nn.sigmoid
    self.output_dir = output_dir

    self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    handles = self.buildGraph()
    self.session.run(tf.initialize_all_variables())
    #self.session.run(tf.contrib.layers.xavier_initializer(uniform=False))
    
    (self.x_in, self.mu_in, self.log_sigma_in, self.z, self.z_mean, 
     self.z_log_sigma, self.x_encoded, self.x_reconstructed, 
     self.rec_cost_mean, self.kl_cost_mean, self.cost, self.global_step, 
     self.train_op) = handles
    
    #self.vgg_net.load_weights(self.session)

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
    CONV_FILTER_SIZES = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    CONV_NUM_CHANNELS = [IMG_DIM['channels'], 64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    POOL_SIZES = [1, 2, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2]
    final_size = (int(IMG_DIM['width'] / np.prod(POOL_SIZES)), 
                  int(IMG_DIM['height'] / np.prod(POOL_SIZES)),
                  CONV_NUM_CHANNELS[-1])
    #FC_SIZES = [int(np.prod(final_size)), 1024, 1024, 1024, 1024, 1024, 1024]
    FC_SIZES = [4096, 1024, 1024, 1024, 1024, 1024, 1024]
    # network inputs
    x_in = tf.placeholder(tf.float32, shape=[None, 4096], 
                          name='x')
    mu_in = tf.placeholder(tf.float32, shape=[None, LATENT_CODE_SIZE])
    log_sigma_in = tf.placeholder(tf.float32, shape=[None, 1])
    dropout = tf.placeholder_with_default(1.0, shape=[], name='dropout')
    mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, 
                       shape=[1, 1, 1, 3], name='img_mean')

    #x_scaled = tf.tanh(x_in)
    with tf.device('/gpu:0'):
      # subtract mean
      nonlinearity_list = [tf.tanh] * (len(FC_SIZES)-1)
      #nonlinearity_list[-1] = tf.identity
      enc_fc_layers = [Dense_NoShare('fc'+str(fc_id), output_size, dropout, non_lin)
		       for (fc_id, output_size, non_lin) in 
                       zip(range(len(FC_SIZES)-1), FC_SIZES[1:], nonlinearity_list)]
      x_encoded = composeAll(reversed(enc_fc_layers))(x_in)

    with tf.device('/gpu:0'):
      # mean and standard deviation for sampling latent code
      z_mean = tf.tanh(Dense_NoShare('z_mean', LATENT_CODE_SIZE)(x_encoded))
      z_log_sigma = tf.tanh(Dense_NoShare('z_log_std', LATENT_CODE_SIZE)(x_encoded))

      # sample latent code
      z = self.sampleGaussian(z_mean, z_log_sigma)

      # project back to VGG fc dimension
      z_proj = tf.nn.tanh(Dense_NoShare('z_proj', FC_SIZES[-1])(z))

      # reconstruction
      nonlinearity_list = [tf.tanh] * (len(FC_SIZES)-1)
      nonlinearity_list[0] = tf.sigmoid
      dec_fc_layers = [Dense_NoShare('fc'+str(fc_id), output_size, dropout, nonlin)
                       for (fc_id, output_size, nonlin) in zip(range(len(FC_SIZES[:-1])), 
                       FC_SIZES[:-1], nonlinearity_list)]
      x_reconstructed = composeAll(dec_fc_layers)(z_proj)
  
      #x_reconstructed = 255.0/2.0 * (composeAll(dec_conv_layers)(z_reshape) + 1.0)
      #x_rec_scaled = 128.0 * tf.nn.softsign(x_reconstructed)
      #x_reconstructed = tf.sigmoid(composeAll(dec_conv_layers)(z_reshape))

      #rec_loss = Model.l2_loss(x_reconstructed, x_in)
      rec_loss = Model.cross_entropy_loss(x_reconstructed, x_in)
      kl_loss = Model.kl_loss(z_mean, z_log_sigma, mu_in, log_sigma_in)
      reg_loss = Model.l2_reg(tf.trainable_variables())

    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('rec_cost_mean'):
      rec_cost_mean = tf.reduce_mean(rec_loss)

    with tf.name_scope('kl_cost_mean'):
      kl_cost_mean = tf.reduce_mean(kl_loss)

    #with tf.name_scope()

    with tf.name_scope('cost'):
      cost = tf.reduce_mean(eval(sys.argv[7]) * rec_loss + 1.0 * kl_loss + 1.0 * reg_loss,
                            name='vae_cost')

    # optimization
    with tf.name_scope('Adam_optimizer'):
      optimizer_coldstart = tf.train.AdamOptimizer(10.0**-eval(sys.argv[5]))
      optimizer_warmstart = tf.train.AdamOptimizer(10.0**-eval(sys.argv[6]))
      tvars = tf.trainable_variables()
      #print tvars
      #for (i, v) in enumerate(tvars):
      #  print v.name, type(v.name), v.device
      #print len(tvars)
      tvars_coldstart = filter(lambda x: x.name.startswith('z'), tvars)
      tvars_warmstart = filter(lambda x: not(x.name.startswith('z')), tvars)
      #for (i, v) in enumerate(tvars):
      #  print v.name, v.device
      #print len(tvars)
      grads_and_vars_coldstart = optimizer_coldstart.compute_gradients(cost, tvars_coldstart, 
                       colocate_gradients_with_ops=True)
      train_op_cs = optimizer_coldstart.apply_gradients(grads_and_vars_coldstart, 
                                 global_step=global_step, name='minimize_cost')
      grads_and_vars_warmstart = optimizer_warmstart.compute_gradients(cost, tvars_warmstart, 
                       colocate_gradients_with_ops=True)
      train_op_ws = optimizer_warmstart.apply_gradients(grads_and_vars_warmstart, 
                                 global_step=global_step, name='minimize_cost')

    return (x_in, mu_in, log_sigma_in, z, z_mean, z_log_sigma, x_encoded,
            x_reconstructed, rec_cost_mean, kl_cost_mean, cost, global_step, train_op_cs)
    
  def train(self, train_data, max_iter=np.inf, max_epochs=np.inf, outdir='./out'):
    saver = tf.train.Saver(tf.all_variables())

    self.latent_codes = {}

    var_inf = VarInf()

    try:
      err_train = 0
      training_start_time = datetime.now()
      now = datetime.now().isoformat()[11:]

      print '------------ Training begin: {} -----------\n'.format(now)

      while True:
        x, x_annot, one_epoch_completed = train_data.get_next_batch()
        mu_true_path = var_inf.get_true_path_mean(x_annot)
        batch_size = np.size(x, 0)
        log_sigma_true_path = np.log(SIGMA_Z) * np.ones(shape=(batch_size, 1))
        feed_dict = {self.x_in: x, 
                     self.mu_in: mu_true_path,
                     self.log_sigma_in: log_sigma_true_path}
        fetches = [self.z, self.z_mean, self.z_log_sigma, self.x_encoded,
                   self.x_reconstructed, self.rec_cost_mean,
                   self.kl_cost_mean, self.cost, self.global_step, self.train_op]

        (z, z_mean, z_log_sigma, x_encoded, x_reconstructed, rec_cost_mean, 
         kl_cost_mean, cost, iteration, _) = self.session.run(fetches, feed_dict)

        if (iteration % 1000 == 0):
          (hist_x, bins_x) = np.histogram(x)
          (hist_xr, bins_xr) = np.histogram(x_reconstructed)
          print hist_x
          print bins_x
          print hist_xr
          print bins_xr

        self.update_latent_codes(z, x_annot)

        # update variational parameters periodically and save current state
        if iteration%10000 == 0:
          saver.save(self.session, os.path.join(self.output_dir, 'model'), 
                     global_step = iteration)
          self.write_latent_codes(os.path.join(self.output_dir, 
                                               'z_'+unicode(iteration)+'.txt'))
          var_inf.update_variational_parameters(self.latent_codes)
          var_inf.write_alpha(os.path.join(self.output_dir, 
                                           'alpha_'+unicode(iteration)+'.txt'))

        err_train += cost
        print (('Iter : %d \t ' +
                'Rec. : %f \t' +
                'KL-div : %f \t' +
                'Loss : %f') % (iteration, rec_cost_mean, kl_cost_mean, cost))

    except KeyboardInterrupt:
        now = datetime.now().isoformat()[11:]
        print '---------- Training end: {} -----------\n'.format(now)
        # write model
        #saver.save(self.session, os.path.join(self.output_dir, 'model'), 
        #           global_step = iteration)
        sys.exit(0)

  def predict(self, data):
    # load model
    saver = tf.train.Saver(tf.all_variables())
    saver.restore(self.session, os.path.join(self.output_dir, 'model-10000'))
    print 'model loaded'
    all_z = np.zeros(shape=(0, 10))
    while True:
      x, x_annot, one_epoch_completed = data.get_next_batch()
      feed_dict = {self.x_in: x}
      fetches = [self.z, self.z_mean, self.z_log_sigma, self.x_encoded,
                 self.x_reconstructed]
      (z, z_mean, z_log_sigma, x_encoded, x_reconstructed) = \
                  self.session.run(fetches, feed_dict)
      all_z = np.append(all_z, z, axis=0)
      #print z_mean
      #print z_log_sigma
      if one_epoch_completed == True:
        break
    print np.shape(all_z)

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
          #print self.latent_codes[vidid][frameid]
          #print type(vidid), vidid
          #print type(frameid), frameid
          #arr = ' '.join(map(str, self.latent_codes[vidid][frameid]))
          #print type(arr), len(arr)
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
      return tf.reduce_mean(tf.square(obs - actual), [1])

  @staticmethod
  def l2_reg(vars):
    loss = 0
    for v in vars:
      loss += tf.nn.l2_loss(v)
    return loss

  @staticmethod
  def cross_entropy_loss(obs, actual):
    #obs_scaled = obs / 255.0
    #actual_scaled = actual / 255.0
    offset = 1E-7
    with tf.name_scope('cross_entropy'):
      obs_ = tf.clip_by_value(obs, offset, 1 - offset)
      return -tf.reduce_mean(actual * tf.log(obs_) + (1 - actual) * tf.log(1 - obs_), [1])
