from util import *
from copy import deepcopy
from scipy.special import digamma

class VarInf(object):
  def __init__(self):
    # initialize variational parameters
    self.alpha = np.random.normal(size=(NUM_NODES, LATENT_CODE_SIZE))
    self.sigmasqr_inv = 1E-6 * np.ones(shape=(NUM_NODES, LATENT_CODE_SIZE))
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
        _ = self.phi[vidid]
      except KeyError:
        self.phi[vidid] = {}
      try:
        _ = self.phi[vidid][frameid]
      except KeyError:
        self.phi[vidid][frameid] = [1.0 / NUM_PATHS] * NUM_PATHS
      path_prob = self.phi[vidid][frameid]
      #path_prob = VarInf.normalize(path_prob)
      #print path_prob
      # sample a path
      true_path_idx = NUM_INTERNAL_NODES + np.random.choice(NUM_PATHS, p=path_prob)
      #print true_path_idx
      #print self.sigmasqr_inv[true_path_idx]
      #print np.diag(1E-4 + 1.0 / self.sigmasqr_inv[true_path_idx])
      # true path params
      true_path_mu = np.random.multivariate_normal(self.alpha[true_path_idx], 
                                      np.diag(1.0 / self.sigmasqr_inv[true_path_idx]))
      true_path_mu_batch.append(np.squeeze(true_path_mu))
    return np.asarray(true_path_mu_batch)

  def update_variational_parameters(self, latent_codes):
    print 'Performing variational inference...'
    print self.alpha
    print self.sigmasqr_inv
    (sum_phi_z, sum_phi) = self.compute_sums(latent_codes)
    print sum_phi
    iteration = 0
    while True:
      iteration += 1
      # save old values
      old_alpha = deepcopy(self.alpha)
      old_sigmasqr_inv = deepcopy(self.sigmasqr_inv)

      # compute new values
      self.compute_alpha_sigma(latent_codes)
      self.compute_gamma(latent_codes)
      self.compute_phi(latent_codes)

      # check for termination
      alpha_diff = np.linalg.norm(self.alpha - old_alpha)
      sigma_diff = np.linalg.norm(self.sigmasqr_inv - old_sigmasqr_inv)
      if (alpha_diff + sigma_diff < 1E-7):
        break
      if (iteration >= 10):
        break
    print 'completed in ' + str(iteration) + ' iterations'
    print self.alpha
    print sum_phi
    print np.asarray(sum_phi) / 161015.0
    print self.sigmasqr_inv
   
  def compute_sums(self, latent_codes):
    sum_phi_z = NUM_PATHS * [0.0]
    sum_phi = NUM_PATHS * [0.0]
    for vidid in latent_codes:
      for frameid in latent_codes[vidid]:
        z_mn = latent_codes[vidid][frameid]
        try:
          phi_mn = self.phi[vidid][frameid]
        except KeyError:
          print 'initializing phi'
          dist2alphas = np.linalg.norm(z_mn - self.alpha[NUM_INTERNAL_NODES:], axis=1)
          phi_mn = np.zeros(shape=NUM_PATHS)
          phi_mn[np.argmin(dist2alphas)] = 1.0
          try:
            _ = self.phi[vidid]
          except KeyError:
            self.phi[vidid] = {}
          self.phi[vidid][frameid] = phi_mn
        for i in range(NUM_PATHS):
          sum_phi_z[i] += phi_mn[i] * z_mn
          sum_phi[i] += phi_mn[i]
    return sum_phi_z, sum_phi

  def compute_alpha_sigma(self, latent_codes):
    # compute sums required
    sum_phi_z, sum_phi = self.compute_sums(latent_codes)
    #SIGMA_Z = self.SIGMA_Z * (1 + 100.0 * np.exp(-global_iter/100.0))
    SIGMA_Z_sqrinv = 1.0 / (SIGMA_Z ** 2)
    for i in reversed(range(NUM_NODES)):
      if i >= NUM_INTERNAL_NODES:
        # leaf node
        parent_i = VarInf.get_parent(i)
        print 'updating sigma', i, sum_phi[i - NUM_INTERNAL_NODES], SIGMA_Z_sqrinv, self.sigmasqr_inv[i], 
        self.sigmasqr_inv[i] = self.sigmasqr_inv[parent_i] +  \
                              sum_phi[i - NUM_INTERNAL_NODES] * SIGMA_Z_sqrinv
        print self.sigmasqr_inv[i]
        self.alpha[i] = (self.sigmasqr_inv[parent_i] * self.alpha[parent_i] +
                        sum_phi_z[i- NUM_INTERNAL_NODES] * SIGMA_Z_sqrinv) \
                        / self.sigmasqr_inv[i]
      else:
        # internal node
        parent_i = VarInf.get_parent(i)
        children_i = VarInf.get_children(i)
        self.sigmasqr_inv[i] = self.sigmasqr_inv[parent_i] if i > 0 else SIGMA_B_sqrinv
        for c in children_i:
          self.sigmasqr_inv[i] += self.sigmasqr_inv[c]
        self.alpha[i] = (self.sigmasqr_inv[parent_i] * self.alpha[parent_i]) if i > 0 else \
                       (SIGMA_B_sqrinv * ALPHA)
        for c in children_i:
          self.alpha[i] += self.sigmasqr_inv[c] * self.alpha[c]
        self.alpha[i] /= self.sigmasqr_inv[i]

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
          #phi_mn = NUM_PATHS * [1.0 / NUM_PATHS]
          phi_mn = np.random.rand(NUM_PATHS)
          phi_mn = phi_mn / np.sum(phi_mn)
        for i in range(NUM_PATHS):
          for j in self.edges_on_path[i]:
            gamma_m[j][0] += phi_mn[i]
          for j in self.edges_before_path[i]:
            gamma_m[j][1] += phi_mn[i]
      self.gamma[vidid] = gamma_m

  def compute_phi(self, latent_codes):
    #SIGMA_Z = self.SIGMA_Z * (1 + 100.0 * np.exp(-global_iter/100.0))
    for vidid in latent_codes:
      phi_m = NUM_PATHS * [0.0]
      try:
        _ = self.gamma[vidid]
      except KeyError:
        self.gamma[vidid] = NUM_EDGES * [[1, GAMMA]]
      for i in range(NUM_PATHS):
        for j in self.edges_on_path[i]:
          phi_m[i] +=  digamma(self.gamma[vidid][j][0]) - \
                       digamma(self.gamma[vidid][j][0] + self.gamma[vidid][j][1])
        for j in self.edges_before_path[i]:
          phi_m[i] +=  digamma(self.gamma[vidid][j][1]) - \
                       digamma(self.gamma[vidid][j][0] + self.gamma[vidid][j][1])
      for frameid in latent_codes[vidid]:
        z_mn = latent_codes[vidid][frameid]
        phi_mn = deepcopy(phi_m)
        norm_z_minus_alpha = np.linalg.norm(z_mn - self.alpha[NUM_INTERNAL_NODES:], axis=1)
        norm_sigma = np.sum(1.0 / self.sigmasqr_inv[NUM_INTERNAL_NODES:], axis=1)
        phi_mn += -1.0 / (2.0 * SIGMA_Z**2) * (norm_z_minus_alpha + norm_sigma)
        try:
          _ = self.phi[vidid]
        except KeyError:
          self.phi[vidid] = {}
        self.phi[vidid][frameid] = VarInf.normalize(np.exp(phi_mn))

  @staticmethod
  def normalize(p):
    #print 'normalize : ', np.shape(p)
    #print np.shape(p)
    p = np.asarray(p)
    s = np.sum(p)
    #print np.shape(p/s)
    if s > 1E-7:
      return p/s
    else:
      return Model.normalize(np.ones(shape=np.shape(p)))

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

  def write_assignments(self,filename):
    with open(filename, 'w') as f:
      for vidid in self.phi:
        for frameid in self.phi[vidid]:
          #print type(vidid), type(frameid), type(np.argmax(phi[vidid][frameid]))
          f.write(vidid + '\t' + frameid + '\t' + \
                  unicode(np.argmax(self.phi[vidid][frameid])) + '\n')

 
