from util import *
from copy import deepcopy
from scipy.special import digamma
from datetime import datetime
import sys
from node import Node
from util import *
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples

# INVARIANT : Run variational inference after every epoch. => Video ids + Frame ids fixed.

class VarInf(object):
  def __init__(self):
    # initialize variational parameters
    #self.alpha = np.random.normal(size=(NUM_NODES, LATENT_CODE_SIZE))
    #self.sigmasqr_inv = (1.0 + BRANCHING_FACTOR) * SIGMA_B_sqrinv * \
    #                     np.ones(shape=(NUM_NODES, LATENT_CODE_SIZE))
    
    self.root = None
    self.decay_coeff = 0.0
    self.vidid_frameid_to_idx = []
    self.vidid_to_idx = []
    self.kmeans_models = {}
    self.kmeans_labels = {}
    self.kmeans_ss = {}

  def get_phi_leaves(self, idx):
    result = {}
    unvisited_nodes = [self.root]
    while len(unvisited_nodes) > 0:
      next_node = unvisited_nodes[0]
      unvisited_nodes = unvisited_nodes[1:]
      if next_node.isLeafNode:
        result[next_node] = next_node.phi[idx][0]
      else:
        unvisited_nodes = unvisited_nodes + next_node.children
    return result

  def get_true_path_mean(self, x_annot_batch):
    true_path_mu_batch = []
    indices = []
    for x_annot in x_annot_batch:
      (vidid, frameid) = x_annot
      if len(self.vidid_frameid_to_idx) == 0:
        # no paths initiated
        true_path_mu_batch.append(np.zeros(shape=LATENT_CODE_SIZE))
      else:
        #idx = self.vidid_frameid_to_idx[vidid][frameid]
        idx = self.vidid_frameid_to_idx.index((vidid, frameid))
        phi = self.get_phi_leaves(idx)
        keys = phi.keys()
        values = VarInf.normalize(phi.values())
        #print idx, map(lambda x:x.node_id, keys), values
        sample_path_idx = np.random.choice(len(values), p=values)
        indices.append(sample_path_idx)
        true_path_mu_batch.append(keys[sample_path_idx].alpha)
    print indices
    return np.asarray(true_path_mu_batch)

  def get_matrix_from_dict(self, latent_codes):
    if len(self.vidid_to_idx) == 0:
      for vidid in latent_codes:
        self.vidid_to_idx.append(vidid)
        for frameid in latent_codes[vidid]:
          self.vidid_frameid_to_idx.append((vidid, frameid))
    result = []
    for vidid in latent_codes:
      for frameid in latent_codes[vidid]:
        result.append(latent_codes[vidid][frameid])
    return np.asarray(result)

  def update_variational_parameters(self, latent_codes):
    print 'Performing variational inference...'
    latent_codes_matrix = self.get_matrix_from_dict(latent_codes)
    if self.root is None:
      self.root = Node('0', len(self.vidid_to_idx), len(self.vidid_frameid_to_idx), None)
      # precompute clusters for different K
      K = 2
      while K < 500:
        kmeans_model = MiniBatchKMeans(n_clusters = K).fit(latent_codes_matrix)
        labels = kmeans_model.labels_
        ss = silhouette_samples(latent_codes_matrix, labels)
        self.kmeans_models[K] = kmeans_model
        self.kmeans_labels[K] = labels
        self.kmeans_ss[K] = ss
        print K
        #print K, ss, np.mean(ss * node.phi)
        K *= 2
    for iteration in range(10):
      print 'Var inf iter ' + str(iteration)
      self.compute_sigma_alpha(latent_codes_matrix)
      print 'sigma alpha computed'
      self.compute_gamma()
      print 'gamma computed'
      self.compute_phi(latent_codes_matrix)
      print 'phi computed'
      self.print_phi(self.root)
      print 'phi printed'
      self.print_gamma(self.root)
      print 'gamma printed'
      raw_input()
      print 'read input'
    split = self.split_nodes(self.root, latent_codes_matrix, \
                             STDEV_THR + 100.0 * np.exp(-self.decay_coeff))
    if split:
      self.decay_coeff = 0.0
    else:
      self.decay_coeff += 1.0

  def print_phi(self, node):
    if node.isLeafNode:
      #print 'phi', node.node_id, np.mean(node.phi, axis=0), node.phi
      print 'phi', node.node_id, np.mean(node.phi, axis=0)
    else:
      for c in node.children:
        self.print_phi(c)

  def print_gamma(self, node):
    if node.isLeafNode:
      if node.gamma is None:
        print 'gamma', node.node_id, node.gamma, node.gamma
      else:
        #print 'gamma', node.node_id, np.mean(node.gamma, axis=0), node.gamma
        print 'gamma', node.node_id, np.mean(node.gamma, axis=0)
    else:
      for c in node.children:
        self.print_gamma(c)

  def split_nodes(self, node, latent_codes_matrix, split_thr):
    if node.isLeafNode:
      # compute variance
      #stdev = np.sqrt(np.linalg.norm(node.phi * latent_codes_matrix - node.alpha) \
      #                / np.sum(node.phi))
      #stdev = np.sqrt(np.linalg.norm(np.sqrt(node.phi) * (latent_codes_matrix - node.alpha)) \
      #                / np.sum(node.phi))
      #print stdev, split_thr
      print node.node_id, np.mean(node.phi), split_thr
      #if stdev > split_thr:
      if np.mean(node.phi) > split_thr:
        #raw_input()
        best_K = -1
        best_K_score = -np.inf
        K = 2
        while K < 500:
          score = np.mean(self.kmeans_ss[K] * node.phi)
          print K, score
          if score > best_K_score:
            best_K, best_K_score = K, score
          K *= 2
        print best_K, best_K_score
        labels = self.kmeans_labels[best_K]
        centers = []
        for l in np.unique(labels):
          centers.append(np.sum(latent_codes_matrix[np.where(labels==l)] * \
                         node.phi[np.where(labels==l)], axis=0) / \
                         np.sum(node.phi[np.where(labels==l)]))
          #print centers
          print np.shape(centers)
        node.isLeafNode = False
        for k in range(best_K):
          new_node = Node(node.node_id + '-' + str(k), \
                          len(self.vidid_to_idx), len(self.vidid_frameid_to_idx), node)
          #new_node.alpha = centers[k]
          new_node.alpha = np.random.normal(size=LATENT_CODE_SIZE)
          new_node.sigmasqr_inv = node.sigmasqr_inv
          new_node.phi = node.phi / best_K
          node.children.append(new_node)
        return True
      #raw_input()
    else:
      result = False
      for c in node.children:
        result = result or self.split_nodes(c, latent_codes_matrix, split_thr)
      return result

  def compute_sigma_alpha_node(self, node, latent_codes_matrix):
    if node.isLeafNode:
      sum_phi = np.sum(node.phi)
      sum_phi_z = np.sum(node.phi * latent_codes_matrix, axis=0)
      node.sigmasqr_inv = SIGMA_B_sqrinv + sum_phi * SIGMA_Z_sqrinv
      try:
        node.alpha = node.parent.alpha
      except AttributeError:
        node.alpha = ALPHA
      node.alpha = (node.alpha * SIGMA_B_sqrinv + sum_phi_z * SIGMA_Z_sqrinv) \
                   / node.sigmasqr_inv
    else:
      # recursively find alpha's and sigma's of children nodes
      #map(self.compute_sigma_alpha_node, node.children)
      for c in node.children:
        self.compute_sigma_alpha_node(c, latent_codes_matrix)
      node.sigmasqr_inv = (1.0 + len(node.children)) * SIGMA_B_sqrinv
      try:
        node.alpha = node.parent.alpha
      except AttributeError:
        node.alpha = ALPHA
      for c in node.children:
        node.alpha += c.alpha
      node.alpha = node.alpha * SIGMA_B_sqrinv / node.sigmasqr_inv

  def compute_sigma_alpha(self, latent_codes_matrix):
    self.compute_sigma_alpha_node(self.root, latent_codes_matrix)

  @staticmethod
  def digamma_add0(gamma_sum, new_gamma):
    return gamma_sum + digamma(new_gamma[:, 0]) - digamma(np.sum(new_gamma, axis=1))

  @staticmethod
  def digamma_add1(gamma_sum, new_gamma):
    if gamma_sum is None:
      return digamma(new_gamma[:, 1]) - digamma(np.sum(new_gamma, axis=1))
    return gamma_sum + digamma(new_gamma[:, 1]) - digamma(np.sum(new_gamma, axis=1))

  def compute_phi_node(self, node, gamma_sum_on, gamma_sum_before, latent_codes_matrix):
    # gamma_sum_on / gamma_sum_before : NUM_VIDS
    # add contribution from parent edge
    if not(node.parent is None):
      gamma_sum_on = VarInf.digamma_add0(gamma_sum_on, node.gamma)
    if node.isLeafNode:
      scaled_dist = 0.5 * SIGMA_Z_sqrinv * \
                    (np.linalg.norm(latent_codes_matrix - node.alpha, axis=1) + \
                     1.0 / node.sigmasqr_inv)
      for i in range(len(self.vidid_frameid_to_idx)):
        vidid, frameid = self.vidid_frameid_to_idx[i]
        #print np.shape(gamma_sum_on[self.vidid_to_idx.index(vidid)])
        #print np.shape(gamma_sum_before[self.vidid_to_idx.index(vidid)])
        #print np.shape(scaled_dist[i])
        #print i, np.shape(node.phi[i]),
        node.phi[i] = gamma_sum_on[self.vidid_to_idx.index(vidid)] + \
                      gamma_sum_before[self.vidid_to_idx.index(vidid)] + \
                      scaled_dist[i]
        #print np.shape(node.phi[i])
      #return VarInf.digamma_add1(gamma_sum_before, node.gamma)
      if node.gamma is None:
        return None
      else:
        return VarInf.digamma_add1(None, node.gamma)
    else:
      gamma_sum_children = np.zeros(shape=(len(self.vidid_to_idx)))
      for c in node.children:
        gamma_sum_children += \
            self.compute_phi_node(c, gamma_sum_on, gamma_sum_before + gamma_sum_children, \
                                  latent_codes_matrix)
      if node.gamma is None:
        return None
      else:
        return VarInf.digamma_add1(gamma_sum_before + gamma_sum_children, node.gamma)

  def get_phi_min(self, node):
    if node.isLeafNode:
      return node.phi
    else:
      phi_min = -np.inf * np.ones(shape=(len(self.vidid_frameid_to_idx), 1))
      for c in node.children:
        phi_min = np.maximum(phi_min, self.get_phi_min(c))
      return phi_min

  def get_phi_sum(self, node):
    #print node.node_id
    if node.isLeafNode:
      #print 'leaf'
      #print node.node_id
      #print node.phi
      return node.phi
    else:
      #print 'internal'
      #print node.children
      #print map(lambda x:x.node_id, node.children)
      phi_sum = np.zeros(shape=(len(self.vidid_frameid_to_idx), 1))
      for c in node.children:
        #tmp = self.get_phi_sum(c)
        #print np.shape(tmp), np.shape(phi_sum), tmp
        phi_sum += self.get_phi_sum(c)
      return phi_sum

  def normalize_phi(self, node, phi_sum):
    if node.isLeafNode:
      node.phi /= phi_sum
      #print node.node_id
      #print node.phi
    else:
      for c in node.children:
        self.normalize_phi(c, phi_sum)

  def exponentiate_phi(self, node, offset):
    if node.isLeafNode:
      #print node.node_id
      #print 'before exp', node.phi
      #print 'offset', offset
      node.phi = np.exp((node.phi - offset))
      #print 'after exp', node.phi
      #raw_input()
    else:
      for c in node.children:
        self.exponentiate_phi(c, offset)

  def compute_phi(self, latent_codes_matrix):
    self.compute_phi_node(self.root, np.zeros(shape=len(self.vidid_to_idx)), \
                          np.zeros(shape=len(self.vidid_to_idx)), latent_codes_matrix)
    phi_min = self.get_phi_min(self.root)
    self.exponentiate_phi(self.root, phi_min)
    phi_sum = self.get_phi_sum(self.root)
    #print 'phi_sum before'
    #print phi_sum
    self.normalize_phi(self.root, phi_sum)
    phi_sum = self.get_phi_sum(self.root)
    #print 'phi_sum after'
    #print phi_sum
    #raw_input()

  def compute_gamma_node(self, node, sum_phi_before):
    '''
    if node.parent is None:
      node.gamma = None
      return
    '''
    if node.isLeafNode:
      if node.parent is None:
        node.gamma = None
        return None
      sum_phi_curr = np.zeros(shape=len(self.vidid_to_idx))
      for i in range(len(self.vidid_to_idx)):
        vidid = self.vidid_to_idx[i]
        phi_vidid = filter(lambda idx: self.vidid_frameid_to_idx[idx][0]==vidid, \
                           range(len(self.vidid_frameid_to_idx)))
        sum_phi_curr[i] = np.sum(node.phi[phi_vidid])
      node.gamma[:, 0] = 1.0 + sum_phi_curr
      node.gamma[:, 1] = GAMMA + sum_phi_before
      #print 'compute_gamma_node', node.node_id, sum_phi_curr, sum_phi_before
      return sum_phi_curr
    else:
      sum_phi_children = np.zeros(shape=len(self.vidid_to_idx))
      for c in node.children:
        sum_phi_children += self.compute_gamma_node(c, sum_phi_before + sum_phi_children)
      return sum_phi_before + sum_phi_children

  def compute_gamma(self):
    self.compute_gamma_node(self.root, np.zeros(shape=len(self.vidid_to_idx)))

  @staticmethod
  def normalize(p):
    p = np.asarray(p)
    s = np.sum(p)
    if s > 0:
      return p/s
    else:
      return VarInf.normalize(np.ones(shape=np.shape(p)))

  def write_gamma(self, filename):
    with open(filename, 'w') as f:
      for vidid in self.gamma:
        f.write(vidid + '\t' + unicode(self.gamma[vidid]) + '\n')

  def write_assignments(self, filename):
    with open(filename, 'w') as f:
      for vidid in self.phi:
        for frameid in self.phi[vidid]:
          f.write(vidid + '\t' + frameid + '\t' + \
                  unicode(np.argmax(self.phi[vidid][frameid])) + '\n')

  def write_alpha_node(self, node, fileptr):
    fileptr.write(node.node_id + ' ' + \
                  ' '.join(map(lambda x: str(np.round(x, 2)), node.alpha)) + \
                  '\n')
    if not(node.isLeafNode):
      for c in node.children:
        self.write_alpha_node(c, fileptr)

  def write_alpha(self, filename):
    fileptr = open(filename, 'w')
    self.write_alpha_node(self.root, fileptr)
    fileptr.close()
      
  def write_sigma_node(self, node, fileptr):
    fileptr.write(node.node_id + ' ' + \
                  ' '.join(map(lambda x: str(np.round(x, 2)), [node.sigmasqr_inv])) + \
                  '\n')
    if not(node.isLeafNode):
      for c in node.children:
        self.write_sigma_node(c, fileptr)

  def write_sigma(self, filename):
    fileptr = open(filename, 'w')
    self.write_sigma_node(self.root, fileptr)
    fileptr.close()
      
