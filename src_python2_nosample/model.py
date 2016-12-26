from __future__ import division
from __future__ import with_statement
from __future__ import absolute_import
from datetime import datetime
import sys
import json
import numpy as np
import tensorflow as tf

from util import *
from layers import *
from io import open
from itertools import imap

DEFAULT_ARCH = [np.prod(IMG_DIM), 1024, 1024, NUM_PATHS]
DEFAULT_LEARNING_RATE = 1E-3
class Model(object):
  # TODO: Add arguments for architecture, learning rate, etc.
  def __init__(self, batch_size):
    u"""
    if architecture == None:
      self.architecture = DEFAULT_ARCH
    else:
      self.architecture = architecture

    if learning_rate == None:
      self.learning_rate = DEFAULT_LEARNING_RATE
    else:
      self.learning_rate = learning_rate
    """
    self.batch_size = batch_size
    self.architecture = DEFAULT_ARCH
    self.learning_rate = DEFAULT_LEARNING_RATE
    self.nonlinearity = tf.nn.elu
    self.squashing = tf.nn.sigmoid

    # initialize path assignments (map of maps)
    self.path_assignments = {}

    self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    handles = self.buildGraph()
    self.session.run(tf.initialize_all_variables())
    
    (self.x_in, self.ncrp_prior, self.theta_normalized, self.z, self.x_reconstructed, 
        self.rec_cost_mean, self.kl_cost_mean, self.cost, self.global_step, 
        self.train_op, self.embedding) = handles

    u'''
    merged_summary_op = tf.merge_all_summaries()
    with tf.Session() as sess:
      sess.run(init)
      writer = tf.train.SummaryWriter("./tf_logs", sess.graph_def)
      writer.close()
    '''

  def sampleMultinomial(self, theta_normalized):
    with tf.name_scope(u"sample_multinomial"):
      tmp = tf.reduce_mean(theta_normalized, axis=1, keep_dims=True, name=u"tmp")
      epsilon = tf.random_uniform(tf.shape(tmp), name=u"epsilon")
      theta_cumulative = tf.cumsum(theta_normalized, axis=1, 
                          name=u"theta_cumulative")
      excess_prob = tf.sub(theta_cumulative, epsilon, name=u"excess_prob")
      excess_prob_rounded = tf.sign(excess_prob, name=u"excess_prob_rounded")
      sample_idx = tf.argmax(excess_prob_rounded, 1, name=u"sample_idx")
      #sample_idx_reshaped = tf.expand_dims(sample_idx, 1, name="sample_idx_reshaped")
      return sample_idx
      #return tf.to_float(sample_idx_reshaped)

  def buildGraph(self):
    x_in = tf.placeholder(tf.float32, shape=[None, self.architecture[0]], name=u"x")
    ncrp_prior = tf.placeholder(tf.float32, shape=[None, self.architecture[-1]], 
                 name=u"ncrp_prior")
    dropout = tf.placeholder_with_default(1., shape=[], name=u"dropout")

    embedding =  tf.Variable(tf.random_normal(([NUM_PATHS, 10])))

    encoding = [Dense(u"encoding", hidden_size, dropout, self.nonlinearity)
                for hidden_size in reversed(self.architecture[1:-1])]
    h_encoded = composeAll(encoding)(x_in)

    log_theta_unnormalized = Dense(u"log_theta_unnormalized", 
                                    self.architecture[-1], dropout)(h_encoded)
    theta_normalized = tf.nn.softmax(log_theta_unnormalized)
    z = self.sampleMultinomial(theta_normalized)

    embed_z = tf.nn.embedding_lookup(embedding, z, name=u"embed_z")

    decoding = [Dense(u"decoding", hidden_size, dropout, self.nonlinearity)
                for hidden_size in self.architecture[1:-1]]
    decoding.insert(0, Dense(u"x_decoding", self.architecture[0], dropout, self.squashing))
    x_reconstructed = tf.identity(composeAll(decoding)(embed_z), name=u"x_reconstructed")

    rec_loss = Model.l2_loss(x_reconstructed, x_in)
    kl_loss = Model.kl_loss(theta_normalized, ncrp_prior)

    with tf.name_scope(u"rec_cost_mean"):
      rec_cost_mean = tf.reduce_mean(rec_loss)

    with tf.name_scope(u"kl_cost_mean"):
      kl_cost_mean = tf.reduce_mean(kl_loss)

    with tf.name_scope(u"cost"):
      cost = tf.reduce_mean(rec_loss + 0.0 * kl_loss, name=u"vae_cost")

    # optimization
    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope(u"Adam_optimizer"):
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      tvars = tf.trainable_variables()
      grads_and_vars = optimizer.compute_gradients(cost, tvars)
      train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step,
                  name=u"minimize_cost")

    return (x_in, ncrp_prior, theta_normalized, z, x_reconstructed, rec_cost_mean, 
            kl_cost_mean, cost, global_step, train_op, embedding)
    
  def train(self, train_data, max_iter=np.inf, max_epochs=np.inf, outdir=u"./out"):
    saver = tf.train.Saver(tf.all_variables())

    try:
      err_train = 0
      now = datetime.now().isoformat()[11:]

      print u"------------ Training begin: {} -----------\n".format(now)

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
        fetches = [self.x_reconstructed, self.theta_normalized, self.z, self.rec_cost_mean,
                  self.kl_cost_mean, self.cost, self.global_step, self.train_op, 
                  self.embedding]
        (x_reconstructed, theta_normalized, z, rec_cost_mean, kl_cost_mean, cost, 
                  iteration, _, embedding) = self.session.run(fetches, feed_dict)

        #print (np.shape(x))
        #print (np.shape(x_reconstructed))

        if iteration%100 == 0:
          # write x and x_reconstructed to file
          with open(u'data_'+unicode(iteration)+u'.txt', u'w') as f:
            for i in xrange(np.size(x, 0)):
              f.write(u"Sample " + unicode(i) + u'\n')
              for j in xrange(np.size(x,1)):
                f.write(unicode(np.round(x[i][j], decimals=2)) + u'\t' + 
                        unicode(np.round(x_reconstructed[i][j], decimals=2)) + u'\t' + 
                        unicode(np.round(np.absolute(x[i][j] - x_reconstructed[i][j]), 
                            decimals=2)) + u'\n')

        print np.shape(embedding)

        if iteration%10 == 1:
          # write embeddings to file
          with open(u'embedding_'+unicode(iteration)+u'.txt', u'w') as f:
            for i in xrange(np.size(embedding, 0)):
              for j in xrange(np.size(embedding, 1)):
                f.write(unicode(np.round(embedding[i][j], decimals=2)) + u'\t')
              f.write(u'\n')

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

        #print (self.path_assignments)
        if iteration%100 == 0:
          Model.write_z(self.path_assignments, u"assignments_"+unicode(iteration)+u".txt")
        err_train += cost
        print iteration, rec_cost_mean, kl_cost_mean, cost
        #print (np.shape(theta_normalized))

    except KeyboardInterrupt:
        now = datetime.now().isoformat()[11:]
        print u"---------- Training end: {} -----------\n".format(now)
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
      #print (prob, prob_norm)
      return prob_norm

  @staticmethod
  def compute_ncrp_prior(path_freq):
    #print ("compute ncrp", path_freq)
    if len(path_freq) == BRANCHING_FACTOR:
      #s = float(sum(path_freq))
      #return map(lambda x: x/s, path_freq)
      return Model.normalize_prob_crp(path_freq)
    else:
      # create higher level frequencies
      parent_freq = []
      for i in xrange(int(len(path_freq)/BRANCHING_FACTOR)):
        parent_freq.append(sum(path_freq[i*BRANCHING_FACTOR:(i+1)*BRANCHING_FACTOR]))
      # compute probabilities for parents recursively
      parent_prob = Model.compute_ncrp_prior(parent_freq)
      #print (parent_freq, parent_prob)
        
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
      #print (vidid, path_freq, ncrp_priors[vidid])
    return ncrp_priors
        
  @staticmethod
  def write_z(z, filename):
    fileptr = open(filename, u'w')
    for vidid in xrange(500):
      try:
        vid_assignments = z[unicode(vidid)]
      except:
        vid_assignments = {}
      if len(vid_assignments) > 0:
        fileptr.write(unicode(vidid) + u"\t" + 
                      u' '.join(imap(unicode, vid_assignments.values())) + u'\n')
    fileptr.close()

  @staticmethod
  def kl_loss(theta_normalized, ncrp_prior):
    offset = 1e-7
    with tf.name_scope(u"kl_loss"):
      theta_normalized_ = tf.clip_by_value(theta_normalized, offset, 1-offset)
      ncrp_prior_ = tf.clip_by_value(ncrp_prior, offset, 1-offset)
      log_theta_normalized = tf.log(theta_normalized_, name=u"log_theta_normalized")
      log_ncrp_prior = tf.log(ncrp_prior_, name=u"log_ncrp_prior")
      #log_diff = tf.subtract(log_ncrp_prior, log_theta_normalized, name="log_diff")
      log_diff = tf.subtract(log_theta_normalized, log_ncrp_prior, name=u"log_diff")
      return tf.reduce_sum(tf.multiply(theta_normalized_, log_diff), 1)
    

  @staticmethod
  def l2_loss(obs, actual):
    with tf.name_scope(u"l2_loss"):
      return tf.reduce_mean(tf.square(obs - actual), 1)
