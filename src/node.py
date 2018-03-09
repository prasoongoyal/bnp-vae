from copy import deepcopy
from scipy.special import digamma
import sys
import numpy as np

class Node(object):
  def __init__(self, node_id, num_vids, num_frames, parent, LATENT_CODE_SIZE, GAMMA):
    print 'Creating node ', node_id
    self.node_id = node_id
    self.isLeafNode = True
    self.parent = parent
    self.alpha = np.random.normal(size=LATENT_CODE_SIZE)
    self.sigmasqr_inv = 1.0
    self.phi = np.ones(shape=(num_frames, 1))
    self.gamma = np.ones(shape=(num_vids, 2))
    self.gamma[:, 1] *= GAMMA
    self.children = []
    
