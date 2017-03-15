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
#import util
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
     self.z_log_sigma, self.x_reconstructed, 
     self.rec_cost_mean, self.kl_cost_mean, self.cost, self.global_step, 
     self.train_op, self.grads, self.valid_data, self.valid_rec, self.t1, 
     self.t2, self.t3) = handles
    
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
    #FC_SIZES = [4096, 1024, 1024, 1024, 1024, 1024, 1024]
    #FC_SIZES = [4096, 1024, 256, 64]
    FC_SIZES = [4096, LATENT_CODE_SIZE]
    # network inputs
    x_in = tf.placeholder(tf.float32, shape=[None, 4096], 
                          name='x')
    mu_in = tf.placeholder(tf.float32, shape=[None, LATENT_CODE_SIZE])
    log_sigma_in = tf.placeholder(tf.float32, shape=[None, 1])
    dropout = tf.placeholder_with_default(1.0, shape=[], name='dropout')
    
    valid_data = tf.placeholder(tf.float32, shape=[None, 4096], name='valid_x')
    #mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, 
    #                   shape=[1, 1, 1, 3], name='img_mean')

    global_step = tf.Variable(0, trainable=False)
    #x_scaled = tf.tanh(x_in)
    with tf.device('/gpu:0'):
      # subtract mean
      #nonlinearity_list = [tf.nn.tanh] * (len(FC_SIZES))
      nonlinearity_list_m = [tf.identity] * (len(FC_SIZES)-1)
      nonlinearity_list_m[-1] = tf.nn.softsign
      nonlinearity_list_s = [tf.identity] * (len(FC_SIZES)-1)
      nonlinearity_list_s[-1] = tf.nn.softsign
      enc_fc_layers_m = [Dense_NoShare('fc_m'+str(fc_id), output_size, dropout, non_lin)
		         for (fc_id, output_size, non_lin) in 
                         zip(range(len(FC_SIZES)-1), FC_SIZES[1:], nonlinearity_list_m)]
      enc_fc_layers_s = [Dense_NoShare('fc_s'+str(fc_id), output_size, dropout, non_lin)
		         for (fc_id, output_size, non_lin) in 
                         zip(range(len(FC_SIZES)-1), FC_SIZES[1:], nonlinearity_list_s)]
      #enc_fc_layers = [Dense_NoShare('fc'+str(fc_id), output_size, dropout, non_lin)
      #		        for (fc_id, output_size, non_lin) in 
      #                 zip(range(len(FC_SIZES)-1), FC_SIZES[1:], nonlinearity_list)]
      #x_encoded = composeAll(reversed(enc_fc_layers))(x_in)
      #z_mean = composeAll(reversed(enc_fc_layers_m))(x_in)
      #z_log_sigma = composeAll(reversed(enc_fc_layers_s))(x_in)

      # mean and standard deviation for sampling latent code
      z_mean = composeAll(reversed(enc_fc_layers_m))(x_in)
      z_log_sigma = composeAll(reversed(enc_fc_layers_s))(x_in)
      #z_mean = tf.tanh(Dense_NoShare('z_mean', LATENT_CODE_SIZE)(composeAll(reversed(enc_fc_layers_m))(x_in)))
      #z_log_sigma = tf.tanh(Dense_NoShare('z_log_std', LATENT_CODE_SIZE)(composeAll(reversed(enc_fc_layers_s))(x_in)))

      # sample latent code
      z = self.sampleGaussian(z_mean, z_log_sigma, global_step)

      # project back to VGG fc dimension
      #z_proj_layer = Dense_NoShare('z_proj', FC_SIZES[-1], 1.0, tf.nn.softsign)
      #z_proj = (z_proj_layer)(z)
      #z_proj = tf.identity(x_encoded)
      #z_proj = tf.identity(z_mean)

      # reconstruction
      nonlinearity_list = [tf.nn.tanh] * (len(FC_SIZES)-1)
      nonlinearity_list[0] = tf.sigmoid
      dec_fc_layers = [Dense_NoShare('fc'+str(fc_id), output_size, dropout, nonlin)
                       for (fc_id, output_size, nonlin) in zip(range(len(FC_SIZES[:-1])), 
                       FC_SIZES[:-1], nonlinearity_list)]
      x_reconstructed = composeAll(dec_fc_layers)(z)
      #bias1 = tf.cast(tf.Variable(1, trainable=True, name='b1'), tf.float32)
      #bias2 = tf.cast(tf.Variable(1, trainable=True, name='b2'), tf.float32)
      #enc_fc = Dense_NoShare('enc_fc', LATENT_CODE_SIZE, 1.0, tf.nn.softsign)
      #dec_fc = Dense_NoShare('dec_fc', 4096, 1.0, tf.sigmoid)
      #bias1 = tf.Variable(tf.zeros([LATENT_CODE_SIZE]), trainable=True, name='b1')
      #bias2 = tf.Variable(tf.zeros([4096]), trainable=True, name='b2')
      #z_proj = (enc_fc)(x_in)
      #x_reconstructed = (dec_fc)(z_proj)

      #z_proj = x_in + bias1
      #x_reconstructed = z_proj + bias2

      v_mean = composeAll(reversed(enc_fc_layers_m))(valid_data, 1.0)
      v_lgs = composeAll(reversed(enc_fc_layers_s))(valid_data, 1.0)
      v = self.sampleGaussian(v_mean, v_lgs, global_step)
      #v_rec = composeAll(dec_fc_layers)((z_proj_layer)(v, 1.0), 1.0)
      v_rec = composeAll(dec_fc_layers)(v, 1.0)

      #v_proj = valid_data + bias1
      #v_rec = v_proj + bias2
      #v_proj = (enc_fc)(valid_data)
      #v_rec = (dec_fc)(v_proj)
      #v = composeAll(reversed(enc_fc_layers))(valid_data)
      #v_rec = composeAll(dec_fc_layers)(v)
 
      #x_reconstructed = 255.0/2.0 * (composeAll(dec_conv_layers)(z_reshape) + 1.0)
      #x_rec_scaled = 128.0 * tf.nn.softsign(x_reconstructed)
      #x_reconstructed = tf.sigmoid(composeAll(dec_conv_layers)(z_reshape))

      #rec_loss = Model.l2_loss(x_reconstructed, x_in)
      rec_loss = Model.l2_loss(x_reconstructed, x_in)
      sigma_coeff = 1.0 + tf.train.exponential_decay(100.0, global_step, 1000, 0.5, staircase=True)
      kl_loss, t1, t2, t3 = Model.kl_loss(z_mean, z_log_sigma, mu_in, log_sigma_in)
      #kl_loss = Model.emd_loss(z_mean, z_log_sigma, mu_in, log_sigma_in)
      reg_loss = Model.l2_reg(tf.trainable_variables())


    with tf.name_scope('rec_cost_mean'):
      rec_cost_mean = tf.reduce_mean(rec_loss)

    '''
    with tf.name_scope('kl_cost_mean'):
      kl_cost_mean = tf.reduce_mean(kl_loss)
    '''
    #with tf.name_scope()

    with tf.name_scope('cost'):
      '''
      cost = tf.reduce_mean(1000.0 * rec_loss + \
             0.0/100000.0 * tf.cast(tf.minimum(global_step, 10000), tf.float32) * kl_loss + \
                            0.0 * reg_loss,
                            name='vae_cost')
      '''
      kl_coeff = 1.0 - tf.train.exponential_decay(1.0, global_step, 1000, \
                 0.95, staircase=True)
      cost = tf.reduce_mean(1.0 * rec_loss + eval(sys.argv[4]) * kl_loss,
                            name='vae_cost')
      
    # optimization
    with tf.name_scope('Adam_optimizer'):
      #optimizer_coldstart = tf.train.AdamOptimizer(10.0**-eval(sys.argv[6]))
      learning_rate = tf.train.exponential_decay(10**-2.0, global_step, 1000,
                      0.98, staircase=True)
      optimizer_coldstart = tf.train.RMSPropOptimizer(learning_rate)
      #optimizer_warmstart = tf.train.AdamOptimizer(10.0**-eval(sys.argv[6]))
      tvars = tf.trainable_variables()
      print tvars
      #for (i, v) in enumerate(tvars):
      #  print v.name, type(v.name), v.device
      #print len(tvars)
      tvars_coldstart = tvars
      #tvars_coldstart = filter(lambda x:(x.name.startswith('z')), tvars)
      #tvars_warmstart = filter(lambda x: not(x.name.startswith('z')), tvars)
      #for (i, v) in enumerate(tvars_coldstart):
      #  print v.name, v.device
      #raw_input()
      #print len(tvars)
      grads_and_vars_coldstart = optimizer_coldstart.compute_gradients(cost, tvars_coldstart, 
                       colocate_gradients_with_ops=True)
      for (g, v) in grads_and_vars_coldstart:
        print g, v.name
        #print g.name, v.name
      #raw_input()
      clipped = [(tf.clip_by_value(grad, -0.01, 0.01), tvar) # gradient clipping
                 for grad, tvar in grads_and_vars_coldstart]
      grads = map(lambda (x, y): x, clipped)
      train_op_cs = optimizer_coldstart.apply_gradients(clipped, 
                                 global_step=global_step, name='minimize_cost')
      #grads_and_vars_warmstart = optimizer_warmstart.compute_gradients(cost, tvars_warmstart, 
      #                 colocate_gradients_with_ops=True)
      #train_op_ws = optimizer_warmstart.apply_gradients(grads_and_vars_warmstart, 
      #                           global_step=global_step, name='minimize_cost')

    #return (x_in, mu_in, log_sigma_in, z, z_mean, z_log_sigma,
    #        x_reconstructed, rec_cost_mean, tf.Variable(0), cost, global_step, train_op_cs,
    #        grads, valid_data, v_rec, tf.Variable(0), tf.Variable(0), tf.Variable(0))
    return (x_in, mu_in, log_sigma_in, z, z_mean, z_log_sigma,
            x_reconstructed, rec_cost_mean, kl_loss, cost, global_step, train_op_cs,
            grads, valid_data, v_rec, t1, t2, t3)
    #return (x_in, mu_in, log_sigma_in, z, z_mean, z_log_sigma,
    #        x_reconstructed, rec_cost_mean, kl_cost_mean, cost, global_step, train_op_cs,
    #        grads, valid_data, v_rec)
    
  def train(self, train_data, max_iter=np.inf, max_epochs=np.inf, outdir='./out'):
    saver = tf.train.Saver(tf.all_variables())
    tvars = tf.all_variables()
    for v in tvars:
      print v.name
    #raw_input()

    self.latent_codes = {}

    var_inf = VarInf()

    # load validation set
    valid_data = np.load('/work/ans556/prasoon/Features/full_data_valid_th.npy')

    try:
      err_train = 0
      training_start_time = datetime.now()
      now = datetime.now().isoformat()[11:]

      print '------------ Training begin: {} -----------\n'.format(now)

      num_epochs = 0
      iteration = 0
      while True:
        if (iteration % 1000 == 999):
          valid_size = 32
        else:
          valid_size = 8
        x, x_annot, epoch_completed = train_data.get_next_batch()
        mu_true_path = var_inf.get_true_path_mean(x_annot)
        batch_size = np.size(x, 0)
        log_sigma_true_path = np.log(SIGMA_Z) * np.ones(shape=(batch_size, 1))
        feed_dict = {self.x_in: x, 
                     self.mu_in: mu_true_path,
                     self.log_sigma_in: log_sigma_true_path,
                     self.valid_data: valid_data[-valid_size:, :]}
        #feed_dict = {self.x_in: x, 
        #             self.mu_in: np.zeros(shape=(batch_size, LATENT_CODE_SIZE)),
        #             self.log_sigma_in: np.zeros(shape=(batch_size, 1)),
        #             self.valid_data: valid_data[-valid_size:, :]}
        fetches = [self.z, self.z_mean, self.z_log_sigma, 
                   self.x_reconstructed, self.rec_cost_mean,
                   self.kl_cost_mean, self.cost, self.global_step, self.train_op, self.grads,
                   self.valid_rec, self.t1, self.t2, self.t3]

        (z, z_mean, z_log_sigma, x_reconstructed, rec_cost_mean, 
         kl_cost_mean, cost, iteration, _, grads, valid_rec, t1, t2, t3) = \
                    self.session.run(fetches, feed_dict)
        '''
        for g in grads:
          if np.mean(g) == 0.0:
            continue
          print np.shape(g), np.min(np.absolute(g)), np.mean(np.absolute(g)), \
                np.max(np.absolute(g))
          if np.max(np.absolute(g)) > 1.0:
            print 'large gradient'
            raw_input()
        '''
        if (iteration % 1 == 0):
          #bins, freq = np.histogram(t2)
          print iteration, Model.get_correct(x, x_reconstructed), \
                           Model.get_correct(valid_data[-valid_size:, :], valid_rec)
          #                 np.min(t1), np.min(t2), np.min(t3), \
          #                 np.max(t1), np.max(t2), np.max(t3), \
          #                 np.size(t1), np.size(t2), np.size(t3), \
          #                 np.min(z_mean), np.max(z_mean), \
          #                 np.min(z_log_sigma), np.max(z_log_sigma), \
          #                 Model.grad_norm(grads), epoch_completed
          print 'Memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                           #np.linalg.norm(t1), np.linalg.norm(t2), np.linalg.norm(t3)
        self.update_latent_codes(z, x_annot)

        # update variational parameters periodically and save current state
        #if iteration%1000 == 0:
        if epoch_completed:
          num_epochs += 1
          if num_epochs == 1:
            num_epochs = 0
          if num_epochs == 0:
            var_inf.update_variational_parameters(self.latent_codes)
            saver.save(self.session, os.path.join(self.output_dir, 'model'), 
                       global_step = iteration)
            self.write_latent_codes(os.path.join(self.output_dir, 
                                                 'z_'+unicode(iteration)+'.txt'))
            var_inf.write_alpha(os.path.join(self.output_dir, 
                                             'alpha_'+unicode(iteration)+'.txt'))
            var_inf.write_sigma(os.path.join(self.output_dir, 
                                             'sigma_'+unicode(iteration)+'.txt'))
            var_inf.write_nodes(os.path.join(self.output_dir, 
                                             'nodes_'+unicode(iteration)+'.dat'))
        '''
        err_train += cost
        print (('Iter : %d \t ' +
                'Rec. : %f \t' +
                'KL-div : %f \t' +
                'Loss : %f') % (iteration, rec_cost_mean, kl_cost_mean, cost))
        '''

    except KeyboardInterrupt:
        now = datetime.now().isoformat()[11:]
        print '---------- Training end: {} -----------\n'.format(now)
        # write model
        #saver.save(self.session, os.path.join(self.output_dir, 'model'), 
        #           global_step = iteration)
        sys.exit(0)

  @staticmethod
  def grad_norm(grads):
    norm = 0.0
    for g in grads:
      norm += np.linalg.norm(g)
    return norm

  @staticmethod
  def get_correct(x, x_rec):
    #(hist_x, bins_x) = np.histogram(x)
    #(hist_xr, bins_xr) = np.histogram(x_reconstructed)
    #print hist_x
    #print bins_x
    #print hist_xr
    #print bins_xr
    x_ = x.flatten()
    x_rec_ = np.round(x_rec.flatten(), 2)
    #h, e = np.histogram(x_rec_, bins=4)
    #print h, e
    m = np.percentile(x_rec_, 75)
    #m = 1.3 * np.mean(x_rec_)
    x_rec_[x_rec_ < m] = 0.0
    x_rec_[x_rec_ > m] = 1.0
    
    x_diff = np.absolute(x_ - x_rec_)
    total_match = len(np.where(x_diff < 0.5)[0]) * 100.0 / len(x_diff)
    match0 = (len(set.intersection(set(np.where(x_==0)[0]), set(np.where(x_rec_==0)[0]))) \
              * 100.0) / len(np.where(x_==0)[0])
    match1 = (len(set.intersection(set(np.where(x_==1)[0]), set(np.where(x_rec_==1)[0]))) \
              * 100.0) / len(np.where(x_==1)[0])
    return (format(total_match, '.2f'), format(match0, '.2f'), format(match1, '.2f'))

  def predict(self, data):
    # load model
    saver = tf.train.Saver(tf.all_variables())
    saver.restore(self.session, sys.argv[8])
    print 'model loaded'
    all_z = np.zeros(shape=(0, 2 + LATENT_CODE_SIZE))
    while True:
      x, x_annot, epochs_completed = data.get_next_batch()
      feed_dict = {self.x_in: x}
      fetches = [self.z, self.z_mean, self.z_log_sigma,
                 self.x_reconstructed]
      (z, z_mean, z_log_sigma, x_reconstructed) = \
                  self.session.run(fetches, feed_dict)
      all_z = np.append(all_z, np.concatenate((x_annot, z), axis=1), axis=0)
      #print z_mean
      #print z_log_sigma
      if epochs_completed > 0:
        break
    print np.shape(all_z)
    with open(os.path.join(self.output_dir, sys.argv[9]), 'w') as f:
      for z in all_z:
        f.write(z[0] + '\t' + z[1] + '\t' + ' '.join(z[2:]) + '\n')
    #np.savetxt('/work/ans556/output_z.txt', all_z)

  def update_latent_codes(self, z_batch, x_annot_batch):
    for (z, x_annot) in zip(z_batch, x_annot_batch):
      (vidid, frameid) = x_annot
      try:
        latent_codes_vid = self.latent_codes[vidid]
      except KeyError:
        latent_codes_vid = {}
      latent_codes_vid[frameid] = z
      self.latent_codes[vidid] = latent_codes_vid

  def sampleGaussian(self, mu, log_sigma, global_step):
    # (Differentiably!) draw sample from Gaussian with given shape, 
    # subject to random noise epsilon
    with tf.name_scope("sample_gaussian"):
      # reparameterization trick
      #sigma_coeff = 1.0 - tf.train.exponential_decay(1.0, global_step, 1000, \
      #              eval(sys.argv[8]), staircase=True)
      epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
      return mu + 1.0 * epsilon * tf.exp(log_sigma) # N(mu, I * sigma**2)

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
  def emd_loss(mu_pred, log_sigma_pred, mu_in, log_sigma_in):
    # offset = 1e-7
    with tf.name_scope('emd_loss'):
      emd_loss = tf.square(mu_pred - mu_in) + \
                 tf.square(tf.exp(log_sigma_pred) - tf.exp(log_sigma_in))
      return tf.reduce_sum(emd_loss, axis=1)

  @staticmethod
  def kl_loss(mu_pred, log_sigma_pred, mu_in, log_sigma_in):
    # offset = 1e-7
    with tf.name_scope('kl_loss'):
      kl_loss = 2.0 * tf.reduce_sum(tf.subtract(log_sigma_in, log_sigma_pred), axis=1)
      mu_diff_scaled = tf.div(tf.subtract(mu_in, mu_pred), tf.exp(log_sigma_in))
      t1 = kl_loss
      kl_loss = tf.add(kl_loss, tf.reduce_sum(tf.square(mu_diff_scaled), axis=1))
      t2 = kl_loss  - t1
      kl_loss = tf.add(kl_loss, tf.reduce_sum(tf.div(tf.exp(2.0 * log_sigma_pred), 
                                       tf.exp(2.0 * log_sigma_in)), axis=1))
      t3 = kl_loss - (t1 + t2)
      #return tf.maximum(eval(sys.argv[7]), kl_loss), t1, t2, t3
      return kl_loss, t1, t2, t3

  @staticmethod
  def l1_loss(obs, actual):
    with tf.name_scope('l2_loss'):
      return tf.reduce_mean(tf.abs(obs - actual), [1])

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
