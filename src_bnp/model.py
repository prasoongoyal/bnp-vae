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
from io import open
from itertools import imap

#DEFAULT_ARCH = [np.prod(IMG_DIM), 1024, 1024, NUM_PATHS]
DEFAULT_LEARNING_RATE = 1E-3
LATENT_CODE_SIZE = 10

class Model(object):
  # TODO: Add arguments for architecture, learning rate, etc.
  def __init__(self, batch_size, output_dir):
    self.batch_size = batch_size
    #self.architecture = DEFAULT_ARCH
    self.learning_rate = DEFAULT_LEARNING_RATE
    self.nonlinearity = tf.nn.elu
    self.squashing = tf.nn.sigmoid
    self.output_dir = output_dir

    # initialize path assignments (map of maps)
    self.path_assignments = {}

    self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    handles = self.buildGraph()
    self.session.run(tf.initialize_all_variables())
    
    (self.x_in, self.mu_in, self.log_sigma_in, self.z, self.x_reconstructed, 
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
    #ncrp_prior = tf.placeholder(tf.float32, shape=[None, NUM_PATHS], 
    #                            name='ncrp_prior')
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

    # probability over paths, to be used in KL-loss
    # path_prob = Dense('path_prob', NUM_PATHS, 1., tf.nn.sigmoid)(x_encoded)

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
    x_reconstructed = tf.sigmoid(composeAll(dec_conv_layers)(z_reshape))

    rec_loss = Model.l2_loss(x_reconstructed, x_in)
    kl_loss = Model.kl_loss(z_mean, z_log_sigma, mu_in, log_sigma_in)

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

    return (x_in, mu_in, log_sigma_in, z, x_reconstructed, rec_cost_mean, 
            kl_cost_mean, cost, global_step, train_op)
    
  def train(self, train_data, max_iter=np.inf, max_epochs=np.inf, outdir='./out'):
    saver = tf.train.Saver(tf.all_variables())

    # ncrp hyperparameters
    #self.ALPHA = np.zeros(shape=(NUM_PATHS, LATENT_CODE_SIZE))
    self.ALPHA = np.zeros(shape=LATENT_CODE_SIZE)
    self.GAMMA = 1.0
    self.SIGMA_B = 1.0
    self.SIGMA_Z = 1.0
    self.SIGMA_B_sqrinv = 1.0 / (self.SIGMA_B ** 2)
    self.SIGMA_Z_sqrinv = 1.0 / (self.SIGMA_Z ** 2)

    # variational parameters
    var_alpha = np.random.normal(size=(NUM_NODES, LATENT_CODE_SIZE))
    var_sigmasqr_inv = np.ones(shape=(NUM_NODES, LATENT_CODE_SIZE))
    var_gamma = {}
    var_phi = {}

    self.latent_codes = {}

    try:
      err_train = 0
      now = datetime.now().isoformat()[11:]

      print '------------ Training begin: {} -----------\n'.format(now)

      # initialize ncrp prior
      ncrp_prior = {}

      while True:
        x, x_annot, one_epoch_completed = train_data.get_next_batch()
        #print np.shape(x_annot)
        mu_true_path = Model.get_true_path_mean(x_annot,
                                              var_alpha, var_sigmasqr_inv, var_gamma, var_phi)
        mu_true_path = np.asarray(mu_true_path)
        batch_size = np.size(x, 0)
        log_sigma_true_path = np.log(self.SIGMA_Z) * np.ones(shape=(batch_size, 1))
        #print mu_true_path[:2,:2]
        #print np.shape(mu_true_path)
        #print np.shape(log_sigma_true_path)
        feed_dict = {self.x_in: x, 
                     self.mu_in: mu_true_path,
                     self.log_sigma_in: log_sigma_true_path}
        fetches = [self.z, self.x_reconstructed, self.rec_cost_mean,
                  self.kl_cost_mean, self.cost, self.global_step, self.train_op]

        (z, x_reconstructed, rec_cost_mean, kl_cost_mean, cost,
         iteration, _) = self.session.run(fetches, feed_dict)

        # self.update_path_assignments(path_prob, x_annot)
        self.update_latent_codes(z, x_annot)

        # write model periodically
        if iteration%1000 == 0:
          saver.save(self.session, os.path.join(self.output_dir, 'model'), 
                     global_step = iteration)
          #Model.write_z(self.path_assignments, os.path.join(self.output_dir,
          #              'assignments'+unicode(iteration)+'.txt'))

        # update variational parameters periodically
        if iteration%10 == 0:
          #print var_alpha
          Model.write_assignments(var_phi, os.path.join(self.output_dir, 
                                           'phi_'+unicode(iteration)+'.json'))
          var_alpha, var_sigmasqr_inv, var_gamma, var_phi = self.update_variational_parameters(
              var_alpha, var_sigmasqr_inv, var_gamma, var_phi)
          #print var_alpha
          #test_dict = {'a': 1, 'b': 2}
          #json.dump(test_dict, open(u'test_dict.json', 'w'))
          #json.dump(test_dict, open(os.path.join(self.output_dir, 
          #          u'phi_' + unicode(iteration) + u'.json'), 'w'))

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
        Model.write_z(self.path_assignments, os.path.join(self.output_dir,
                      'assignments'+unicode(iteration)+'.txt'))
        sys.exit(0)

  @staticmethod
  def write_assignments(phi, filename):
    with open(filename, 'w') as f:
      for vidid in phi:
        for frameid in phi[vidid]:
          #print type(vidid), type(frameid), type(np.argmax(phi[vidid][frameid]))
          f.write(vidid + '\t' + frameid + '\t' + unicode(np.argmax(phi[vidid][frameid])) + '\n')

  def update_variational_parameters(self, var_alpha, var_sigmasqr_inv, var_gamma, var_phi):
    print 'Performing variational inference...'
    #var_sigma = np.exp(var_log_sigma)
    iteration = 0
    edges_on_path = []
    edges_before_path = []
    for i in range(NUM_PATHS):
      edges_on_path.append(Model.get_edges_on_path(i))
      edges_before_path.append(Model.get_edges_before_path(i))
    while True:
      now = datetime.now().isoformat()[11:]
      #print '---------- New iter: \t\t{} -----------'.format(now)
      iteration += 1
      #raw_input()
      #old_var_alpha, old_var_sigma, old_var_gamma, old_var_phi = var_alpha, \
      #    var_sigma, var_gamma, var_phi
      old_var_alpha = deepcopy(var_alpha)
      old_var_sigmasqr_inv = deepcopy(var_sigmasqr_inv)
      #old_var_gamma = deepcopy(var_gamma)
      #old_var_phi = deepcopy(var_phi)
      now = datetime.now().isoformat()[11:]
      #print '---------- Computation started: \t\t{} -----------'.format(now)
      # compute alpha
      sum_phi_z = NUM_PATHS * [0.0]
      sum_phi = NUM_PATHS * [0.0]
      for vidid in self.latent_codes:
        for frameid in self.latent_codes[vidid]:
          z_mn = self.latent_codes[vidid][frameid]
          try:
            phi_mn = var_phi[vidid][frameid]
          except KeyError:
            phi_mn = NUM_PATHS * [1.0 / NUM_PATHS]
          #print np.shape(z_mn)
          #print np.shape(phi_mn)
          for i in range(NUM_PATHS):
            sum_phi_z[i] += phi_mn[i] * z_mn
            sum_phi[i] += phi_mn[i]
      # compute alpha and sigma
      for i in reversed(range(NUM_PATHS)):
        if i >= NUM_INTERNAL_NODES:
          # leaf node
          parent_i = Model.get_parent(i)
          print i, parent_i
          var_sigmasqr_inv[i] = var_sigmasqr_inv[parent_i] +  \
                                sum_phi[i - NUM_INTERNAL_NODES] * self.SIGMA_Z_sqrinv
          var_alpha[i] = (var_sigmasqr_inv[parent_i] * var_alpha[parent_i] + 
                          sum_phi_z[i] * self.SIGMA_Z_sqrinv) / var_sigmasqr_inv[i]
        else:
          # internal node
          parent_i = Model.get_parent(i)
          children_i = Model.get_children(i)
          var_sigmasqr_inv[i] = var_sigmasqr_inv[parent_i] if i > 0 else self.SIGMA_B_sqrinv
          for c in children_i:
            var_sigmasqr_inv[i] += var_sigmasqr_inv[c]
          var_alpha[i] = (var_sigmasqr_inv[parent_i] * var_alpha[parent_i]) if i > 0 else \
                         (self.SIGMA_B_sqrinv * self.ALPHA)
          for c in children_i:
            var_alpha[i] += var_sigmasqr_inv[c] * var_alpha[c]
          var_alpha[i] /= var_sigmasqr_inv[i]
      '''
      for i in range(NUM_PATHS):
        var_alpha[i] = (self.ALPHA[i] + sum_phi_z[i]) / (1.0 + sum_phi[i])
      #print 'alpha updated'
      now = datetime.now().isoformat()[11:]
      #print '---------- a computed: \t\t{} -----------'.format(now)
      # compute sigma
      for i in range(NUM_PATHS):
        var_sigma[i] = self.SIGMA_B / np.sqrt(1.0 + sum_phi[i])
      #print 'sigma updated'
      now = datetime.now().isoformat()[11:]
      #print '---------- s computed: \t\t{} -----------'.format(now)
      '''
      # compute gamma
      for vidid in self.latent_codes:
        try:
          latent_codes_vidid = self.latent_codes[vidid]
        except KeyError:
          latent_codes_vidid = {}
        gamma_m = NUM_EDGES * [[1, self.GAMMA]]
        for frameid in latent_codes_vidid:
          phi_mn = NUM_PATHS * [1.0 / NUM_PATHS]
          for i in range(NUM_PATHS):
            for j in edges_on_path[i]:
              #print i, j, np.shape(gamma_m), np.shape(phi_mn), gamma_m[j]
              gamma_m[j][0] += phi_mn[i]
            for j in edges_before_path[i]:
              gamma_m[j][1] += phi_mn[i]
        var_gamma[vidid] = gamma_m
      #print 'gamma computed'
      now = datetime.now().isoformat()[11:]
      #print '---------- g computed: \t\t{} -----------'.format(now)
      # compute phi
      for vidid in self.latent_codes:
        phi_m = NUM_PATHS * [0.0]
        for i in range(NUM_PATHS):
          for j in edges_on_path[i]:
            phi_m[i] +=  digamma(var_gamma[vidid][j][0]) - \
                         digamma(var_gamma[vidid][j][0] + var_gamma[vidid][j][1])
          for j in edges_before_path[i]:
            phi_m[i] +=  digamma(var_gamma[vidid][j][1]) - \
                         digamma(var_gamma[vidid][j][0] + var_gamma[vidid][j][1])
        for frameid in self.latent_codes[vidid]:
          z_mn = self.latent_codes[vidid][frameid]
          phi_mn = deepcopy(phi_m)
          norm_z_minus_alpha = np.linalg.norm(z_mn - var_alpha[NUM_INTERNAL_NODES:], axis=1)
          norm_sigma = np.sum(1.0 / var_sigmasqr_inv[NUM_INTERNAL_NODES:], axis=1)
          phi_mn += -1.0 / (2.0 * self.SIGMA_Z**2) * (norm_z_minus_alpha + norm_sigma)
          '''
          for i in range(NUM_PATHS):
            phi_mn[i] += -1.0 / (2.0 * self.SIGMA_Z**2) * \
                         (np.linalg.norm(z_mn - var_alpha[i]) + 
                          np.linalg.norm(var_sigma[i]))
          '''
          #print 'before loop : ', np.shape(phi_mn)
            #print 'phi_mn_i : ', phi_mn[i]
          #print 'after loop : ', np.shape(phi_mn)
          try:
            _ = var_phi[vidid]
          except KeyError:
            var_phi[vidid] = {}
          #print vidid, frameid, np.shape(phi_mn)
          var_phi[vidid][frameid] = Model.normalize(np.exp(phi_mn))
      #print 'phi computed'
      now = datetime.now().isoformat()[11:]
      #print '---------- p computed: \t\t{} -----------'.format(now)
      alpha_diff = np.linalg.norm(var_alpha - old_var_alpha)
      sigma_diff = np.linalg.norm(var_sigmasqr_inv - old_var_sigmasqr_inv)
      # print alpha_diff, sigma_diff
      #print 'alphas : ', var_alpha, old_var_alpha
      #print 'sigmas : ', var_sigma, old_var_sigma
      #print 'gammas : ', var_gamma, old_var_gamma
      #print 'phis : ', var_phi, old_var_phi
      if (alpha_diff + sigma_diff < 1E-7):
        break
      if (iteration >= 10):
        break
    print 'completed in ' + str(iteration) + ' iterations'
    return var_alpha, var_sigmasqr_inv, var_gamma, var_phi

  @staticmethod
  def get_parent(i):
    return int((i - 1) / BRANCHING_FACTOR)

  @staticmethod
  def get_children(i):
    return range((BRANCHING_FACTOR * i + 1), (BRANCHING_FACTOR * (i+1)))

  @staticmethod
  def get_edges_on_path(i):
    result = []
    i += NUM_INTERNAL_EDGES
    while (i >= 0):
      result.append(i)
      i = int(i/BRANCHING_FACTOR) - 1
    return result

  @staticmethod
  def get_edges_before_path(i):
    result = []
    i += NUM_INTERNAL_EDGES
    while (i >= 0):
      for j in range(int(i/BRANCHING_FACTOR)*BRANCHING_FACTOR, i):
        result.append(j)
      i = int(i/BRANCHING_FACTOR) - 1
    return result

  def update_latent_codes(self, z_batch, x_annot_batch):
    for (z, x_annot) in zip(z_batch, x_annot_batch):
      (vidid, frameid) = x_annot
      #print vidid, frameid
      try:
        latent_codes_vid = self.latent_codes[vidid]
      except KeyError:
        latent_codes_vid = {}
      latent_codes_vid[frameid] = z
      self.latent_codes[vidid] = latent_codes_vid
    #print self.latent_codes

  def sampleGaussian(self, mu, log_sigma):
    # (Differentiably!) draw sample from Gaussian with given shape, 
    # subject to random noise epsilon
    with tf.name_scope("sample_gaussian"):
      # reparameterization trick
      epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
      return mu + epsilon * tf.exp(log_sigma) # N(mu, I * sigma**2)

  def update_path_assignments(self, path_prob, x_annot):
    z = np.argmax(path_prob, 1)
    for i in xrange(len(z)):
      (vidid, frameid) = x_annot[i]
      try: 
        vid_path_assignments = self.path_assignments[vidid]
      except:
        vid_path_assignments = {}
          
      vid_path_assignments[frameid] = int(z[i])
      self.path_assignments[vidid] = vid_path_assignments

  @staticmethod
  def normalize(p):
    #print 'normalize : ', np.shape(p)
    #print np.shape(p)
    p = np.asarray(p)
    s = np.sum(p)
    #print np.shape(p/s)
    return p/s

  @staticmethod
  def get_true_path_mean(x_annot_batch, alpha, sigmasqr_inv, gamma, phi):
    true_path_mu_batch = []
    for x_annot in x_annot_batch:
      (vidid, frameid) = x_annot
      # get distribution over paths
      try:
        path_prob = phi[vidid][frameid]
      except KeyError:
        path_prob = [1.0 / NUM_PATHS] * NUM_PATHS
      # normalize path prob
      sum_path_prob = float(sum(path_prob))
      path_prob = map(lambda x:x/sum_path_prob, path_prob)
      # sample a path
      true_path_idx = NUM_INTERNAL_NODES + np.random.choice(NUM_PATHS, 1, path_prob)
      # true path params
      true_path_mu = np.random.normal(alpha[true_path_idx], 
                                      np.diag(1.0 / sigmasqr_inv[true_path_idx]))
      true_path_mu_batch.append(np.squeeze(true_path_mu))
    return true_path_mu_batch

  @staticmethod
  def compute_ncrp_prior_batch(x_annot, ncrp_prior):
    ncrp_prior_batch = []
    for (vidid, frameid) in x_annot:
      try:
        ncrp_prior_vid = ncrp_prior[vidid]
      except:
        ncrp_prior_vid = np.ones(NUM_PATHS)
        ncrp_prior_vid = ncrp_prior_vid / np.sum(ncrp_prior_vid)
      ncrp_prior_batch.append(ncrp_prior_vid)
    return ncrp_prior_batch

 
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
  def kl_loss(mu_pred, log_sigma_pred, mu_in, log_sigma_in):
    # offset = 1e-7
    with tf.name_scope('kl_loss'):
      mu_diff = tf.subtract(mu_pred, mu_in, name='mu_diff')
      mu_diff_norm = tf.reduce_sum(tf.square(mu_diff), axis=1)
      sigma_norm = tf.reduce_sum(tf.square(tf.exp(log_sigma_pred)))
      #print type(log_sigma_in)
      #print log_sigma_in.get_shape()
      return tf.subtract(1.0/(tf.exp(log_sigma_in) ** 2) * tf.add(mu_diff_norm, sigma_norm),
                         2.0 * tf.reduce_sum(log_sigma_pred, axis=1))
      '''
      theta_normalized_ = tf.clip_by_value(theta_normalized, offset, 1-offset)
      ncrp_prior_ = tf.clip_by_value(ncrp_prior, offset, 1-offset)
      log_theta_normalized = tf.log(theta_normalized_, name='log_theta_normalized')
      log_ncrp_prior = tf.log(ncrp_prior_, name='log_ncrp_prior')
      log_diff = tf.subtract(log_theta_normalized, log_ncrp_prior, name='log_diff')
      return tf.reduce_sum(tf.multiply(theta_normalized_, log_diff), 1)
      '''
    

  @staticmethod
  def l2_loss(obs, actual):
    with tf.name_scope('l2_loss'):
      return tf.reduce_mean(tf.square(obs - actual), [1, 2, 3])
