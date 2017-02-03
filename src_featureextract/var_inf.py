from util import *
from copy import deepcopy
from scipy.special import digamma
from datetime import datetime
import sys

class VarInf(object):
  def __init__(self):
    # initialize variational parameters
    self.alpha = np.random.normal(size=(NUM_NODES, LATENT_CODE_SIZE))
    self.sigmasqr_inv = (1.0 + BRANCHING_FACTOR) * SIGMA_B_sqrinv * \
                         np.ones(shape=(NUM_NODES, LATENT_CODE_SIZE))
    self.gamma = {}
    self.phi = {}

    self.edges_on_path = []
    self.edges_before_path = []
    for i in range(NUM_PATHS):
      self.edges_on_path.append(VarInf.get_edges_on_path(i))
      self.edges_before_path.append(VarInf.get_edges_before_path(i))

  def get_true_path_mean(self, x_annot_batch):
    true_path_mu_batch = []
    for x_annot in x_annot_batch:
      (vidid, frameid) = x_annot
      # get distributions over path
      try:
        path_prob = self.phi[vidid][frameid]
      except KeyError:
        #path_prob = [1.0 / NUM_PATHS] * NUM_PATHS
        path_prob = np.random.random(size=NUM_PATHS)
        path_prob = path_prob / np.sum(path_prob)
      '''
      try:
        _ = self.phi[vidid]
        mult_factor = np.ones(shape=(NUM_PATHS))
        for f in self.phi[vidid]:
          if (eval(f) < eval(frameid)):
            # reduce probability of highest prob path
            max_prob_path = np.argmax(self.phi[vidid][f])
            for p in range(max_prob_path):
              mult_factor[p] = eval(sys.argv[7])
        path_prob = path_prob * mult_factor
        path_prob = path_prob / np.sum(path_prob)
      except KeyError:
        pass
      '''
      true_path_idx = NUM_INTERNAL_NODES + np.random.choice(NUM_PATHS, p=path_prob)
      true_path_mu = np.random.multivariate_normal(self.alpha[true_path_idx], 
                                      np.diag(1.0 / self.sigmasqr_inv[true_path_idx]))
      #true_path_idx = NUM_INTERNAL_NODES + np.argmax(path_prob)
      #true_path_mu = self.alpha[true_path_idx]
      true_path_mu_batch.append(np.squeeze(true_path_mu))
    return np.asarray(true_path_mu_batch)

  def update_variational_parameters(self, latent_codes):
    print 'Performing variational inference...'
    for iteration in range(1):
      sum_phi_z, sum_phi = self.compute_sums(latent_codes)
      self.compute_sigma(sum_phi_z, sum_phi)
      self.compute_alpha(sum_phi_z, sum_phi)
      self.compute_gamma(latent_codes)
      self.compute_phi(latent_codes)
   
  def compute_sums(self, latent_codes):
    sum_phi_z = NUM_PATHS * [0.0]
    sum_phi = NUM_PATHS * [0.0]
    for vidid in latent_codes:
      for frameid in latent_codes[vidid]:
        z_mn = latent_codes[vidid][frameid]
        try:
          phi_mn = self.phi[vidid][frameid]
        except:
          # initialize phi
          dist2alphas = np.linalg.norm(z_mn - self.alpha[NUM_INTERNAL_NODES:], axis=1)
          phi_mn = np.zeros(shape=NUM_PATHS)
          phi_mn[np.argmin(dist2alphas)] = 1.0
        for i in range(NUM_PATHS):
          sum_phi_z[i] += phi_mn[i] * z_mn
          sum_phi[i] += phi_mn[i]
    return sum_phi_z, sum_phi

  def compute_alpha(self, sum_phi_z, sum_phi):
    for i in reversed(range(NUM_NODES)):
      if i >= NUM_INTERNAL_NODES:
        # leaf node
        parent_i = VarInf.get_parent(i)
        self.alpha[i] = (SIGMA_B_sqrinv * self.alpha[parent_i] +
                        sum_phi_z[i- NUM_INTERNAL_NODES] * SIGMA_Z_sqrinv) \
                        / self.sigmasqr_inv[i]
      else:
        # internal node
        parent_i = VarInf.get_parent(i)
        children_i = VarInf.get_children(i)
        self.alpha[i] = self.alpha[parent_i] if i > 0 else  ALPHA
        for c in children_i:
          self.alpha[i] += self.alpha[c]
        self.alpha[i] /= (1.0 + BRANCHING_FACTOR)

  def compute_sigma(self, sum_phi_z, sum_phi):
    for i in reversed(range(NUM_NODES)):
      if i >= NUM_INTERNAL_NODES:
        # leaf node
        parent_i = VarInf.get_parent(i)
        self.sigmasqr_inv[i] = SIGMA_B_sqrinv +  \
                              sum_phi[i - NUM_INTERNAL_NODES] * SIGMA_Z_sqrinv

  def compute_gamma(self, latent_codes):
    for vidid in latent_codes:
      try:
        latent_codes_vidid = latent_codes[vidid]
      except KeyError:
        latent_codes_vidid = {}
      gamma_m = NUM_EDGES * [[1, GAMMA]]
      for frameid in latent_codes_vidid:
        try:
          phi_mn = self.phi[vidid][frameid]
        except KeyError:
          phi_mn = np.random.rand(NUM_PATHS)
          phi_mn = phi_mn / np.sum(phi_mn)
        for i in range(NUM_PATHS):
          for j in self.edges_on_path[i]:
            gamma_m[j][0] += phi_mn[i]
          for j in self.edges_before_path[i]:
            gamma_m[j][1] += phi_mn[i]
      self.gamma[vidid] = gamma_m

  def compute_phi(self, latent_codes):
    norm_sigma = np.sum(1.0 / self.sigmasqr_inv[NUM_INTERNAL_NODES:], axis=1)
    for vidid in latent_codes:
      phi_m = NUM_PATHS * [0.0]
      try:
        gamma_m = self.gamma[vidid]
      except KeyError:
        gamma_m = NUM_EDGES * [[1, GAMMA]]
      for i in range(NUM_PATHS):
        for j in self.edges_on_path[i]:
          phi_m[i] +=  digamma(gamma_m[j][0]) - \
                       digamma(gamma_m[j][0] + gamma_m[j][1])
        for j in self.edges_before_path[i]:
          phi_m[i] +=  digamma(gamma_m[j][1]) - \
                       digamma(gamma_m[j][0] + gamma_m[j][1])
      for frameid in latent_codes[vidid]:
        z_mn = latent_codes[vidid][frameid]
        norm_z_minus_alpha = np.linalg.norm(z_mn - self.alpha[NUM_INTERNAL_NODES:], axis=1)
        phi_mn = phi_m - 1.0 / (2.0 * SIGMA_Z**2) * (norm_z_minus_alpha + norm_sigma)
        try:
          _ = self.phi[vidid]
        except KeyError:
          self.phi[vidid] = {}
        self.phi[vidid][frameid] = VarInf.normalize(np.exp(phi_mn))

  @staticmethod
  def normalize(p):
    p = np.asarray(p)
    s = np.sum(p)
    if s > 0:
      return p/s
    else:
      return VarInf.normalize(np.ones(shape=np.shape(p)))

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

  def write_alpha(self, filename):
    np.savetxt(filename, self.alpha)
  def write_sigma(self, filename):
    np.savetxt(filename, self.sigmasqr_inv)
      
