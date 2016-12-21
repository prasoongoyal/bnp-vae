from datetime import datetime
import sys
import json
import numpy as np
import tensorflow as tf

from util import *
from layers import *

DEFAULT_ARCH = [np.prod(IMG_DIM), 1024, 1024, NUM_PATHS]
DEFAULT_LEARNING_RATE = 1E-3
class Model():
  # TODO: Add arguments for architecture, learning rate, etc.
  def __init__(self, batch_size):
    """
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
    
    (self.x_in, self.theta_normalized, self.z, self.x_reconstructed, self.cost,
    self.global_step, self.train_op) = handles

    '''
    merged_summary_op = tf.merge_all_summaries()
    with tf.Session() as sess:
      sess.run(init)
      writer = tf.train.SummaryWriter("./tf_logs", sess.graph_def)
      writer.close()
    '''

  def sampleMultinomial(self, theta_normalized):
    with tf.name_scope("sample_multinomial"):
      tmp = tf.reduce_mean(theta_normalized, axis=1, keep_dims=True, name="tmp")
      epsilon = tf.random_uniform(tf.shape(tmp), name="epsilon")
      theta_cumulative = tf.cumsum(theta_normalized, axis=1, 
                          name="theta_cumulative")
      excess_prob = tf.sub(theta_cumulative, epsilon, name="excess_prob")
      excess_prob_rounded = tf.sign(excess_prob, name="excess_prob_rounded")
      sample_idx = tf.argmax(excess_prob_rounded, 1, name="sample_idx")
      sample_idx_reshaped = tf.expand_dims(sample_idx, 1, name="sample_idx_reshaped")
      return tf.to_float(sample_idx_reshaped)

  def buildGraph(self):
    x_in = tf.placeholder(tf.float32, shape=[None, self.architecture[0]], name="x")
    ncrp_prior = tf.placeholder(tf.float32, shape=[None, self.architecture[-1]], 
                 name="ncrp_prior")
    dropout = tf.placeholder_with_default(1., shape=[], name="dropout")

    encoding = [Dense("encoding", hidden_size, dropout, self.nonlinearity)
                for hidden_size in reversed(self.architecture[1:-1])]
    h_encoded = composeAll(encoding)(x_in)

    log_theta_unnormalized = Dense("log_theta_unnormalized", 
                                    self.architecture[-1], dropout)(h_encoded)
    theta_normalized = tf.nn.softmax(log_theta_unnormalized)
    z = self.sampleMultinomial(theta_normalized)

    decoding = [Dense("decoding", hidden_size, dropout, self.nonlinearity)
                for hidden_size in self.architecture[1:-1]]
    decoding.insert(0, Dense("x_decoding", self.architecture[0], dropout, self.squashing))
    x_reconstructed = tf.identity(composeAll(decoding)(z), name="x_reconstructed")

    rec_loss = Model.l2_loss(x_reconstructed, x_in)
    kl_loss = Model.kl_loss(theta_normalized, ncrp_prior)
     
    with tf.name_scope("cost"):
      cost = tf.reduce_mean(tf.add(rec_loss, kl_loss),1)

    # optimization
    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope("Adam_optimizer"):
      optimizer = tf.train.AdamOptimizer(self.learning_rate)
      tvars = tf.trainable_variables()
      grads_and_vars = optimizer.compute_gradients(cost, tvars)
      train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step,
                  name="minimize_cost")

    return (x_in, theta_normalized, z, x_reconstructed, cost, global_step, train_op)
    
  def train(self, train_data, max_iter=np.inf, max_epochs=np.inf, outdir="./out"):
    saver = tf.train.Saver(tf.all_variables())

    try:
      err_train = 0
      now = datetime.now().isoformat()[11:]

      print ("------------ Training begin: {} -----------\n".format(now))

      while True:
        x, x_annot, one_epoch_completed = train_data.get_next_batch()
        feed_dict = {self.x_in: x}
        fetches = [self.x_reconstructed, self.theta_normalized, self.z, self.cost, self.global_step, self.train_op]
        x_reconstructed, theta_normalized, z, cost, iteration, _ = self.session.run(fetches, feed_dict)

        for i in range(len(z)):
          (vidid, frameid) = x_annot[i]
          try: 
            vid_path_assignments = self.path_assignments[vidid]
          except:
            vid_path_assignments = {}
          
          vid_path_assignments[frameid] = int(z[i][0])
          self.path_assignments[vidid] = vid_path_assignments

        #print (self.path_assignments)
        if iteration%100 == 0:
          self.write_z(self.path_assignments, "assignments_"+str(iteration)+".txt")
        err_train += cost
        print (iteration, cost)
        #print (np.shape(theta_normalized))

    except KeyboardInterrupt:
        now = datetime.now().isoformat()[11:]
        print ("---------- Training end: {} -----------\n".format(now))
        sys.exit(0)

  @staticmethod
  def write_z(z, filename):
    fileptr = open(filename, 'w')
    for vidid in range(500):
      try:
        vid_assignments = z[str(vidid)]
      except:
        vid_assignments = {}
      if len(vid_assignments) > 0:
        fileptr.write(str(vidid) + "\t" + 
                      ' '.join(map(str, vid_assignments.values())) + '\n')
    fileptr.close()

  @staticmethod
  def kl_loss(theta_normalized, ncrp_prior):
    with tf.name_scope("kl_loss"):
      log_theta_normalized = tf.log(theta_normalized, name="log_theta_normalized")
      log_ncrp_prior = tf.log(ncrp_prior, name="log_ncrp_prior")
      log_diff = tf.subtract(log_ncrp_prior, log_theta_normalized, name="log_diff")
      return tf.reduce_sum(tf.multiply(theta_normalized, log_diff), 1)
    

  @staticmethod
  def l2_loss(obs, actual):
    with tf.name_scope("l2_loss"):
      return tf.reduce_mean(tf.square(obs - actual), 1)
