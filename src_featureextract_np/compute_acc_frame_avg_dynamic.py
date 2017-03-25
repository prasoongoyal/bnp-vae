import numpy as np
import scipy.stats
import sys
import copy
from node import Node
import cPickle

'''
python compute_acc_frame.py /scratch/ans556/prasoon/all_models/models_test/alpha_140000.txt /scratch/ans556/prasoon/all_models/models_test/z_150000.txt /work/ans556/prediction_test/output_15k_z_train.txt 256 /scratch/ans556/prasoon/data/MED/MED11/train_label.txt /scratch/ans556/prasoon/data/MED/MED11/train_label.txt
'''

def get_alpha_leaves(node, result):
  if node.isLeafNode:
    result[node] = node.alpha
  else:
    for c in node.children:
      result = get_alpha_leaves(c, result)
  return result  

def get_internal_nodes(node, result):
  if node.isLeafNode:
    pass
  else:
    result.append(node)
    for c in node.children:
      result = get_internal_nodes(c, result)
  return result

#def main(nodes_file, z_train_file, z_test_file, bf, num_levels, num_paths, \
#         train_labels_file, test_labels_file, decay_factor):
def main(nodes_filename, train_labels_file, z_train_file, test_labels_file, z_test_file, \
         decay_factor = 0.0):
  nodes_file = open(nodes_filename, 'rb')
  nodes_list = cPickle.load(nodes_file)
  nodes_file.close()
  root_node = None
  for node in nodes_list:
    if node.parent == None:
      root_node = node
      break
  alpha_dict = get_alpha_leaves(root_node, {})
  print len(alpha_dict)
  num_paths = len(alpha_dict)
  leaf_nodes_list = alpha_dict.keys()
  alpha_values = alpha_dict.values()
  train_labels = np.loadtxt(train_labels_file) - 1
  num_classes = len(np.unique(train_labels))
  z_train = np.loadtxt(z_train_file)
  path_assignments = []
  for i, z in enumerate(z_train):
    closest_path_idx = np.argmin(np.linalg.norm((alpha_values - z[2:]), axis=1))
    path_assignments.append(closest_path_idx)
  
  for path_id in range(num_paths):
    frames_assigned = np.where(np.asarray(path_assignments) == path_id)[0]
    # compute the closest 4
    dist_to_path = np.linalg.norm(alpha_values[path_id] - z_train[frames_assigned, 2:], axis=1)
    #print path_id, dist_to_path
    

  counts = np.zeros(shape=(num_paths, num_classes))
  for i, l in enumerate(path_assignments):
    counts[l][int(train_labels[i])] += 1
  #print counts

  counts_cumulative = copy.deepcopy(counts)
  internal_nodes_list = get_internal_nodes(root_node, [])
  num_internal_nodes = len(internal_nodes_list)
  #print num_internal_nodes
  counts_cumulative = np.concatenate((np.zeros(shape=(num_internal_nodes, num_classes)), \
                                     counts_cumulative), axis=0)
  leaf_nodes_id_list = map(lambda x: x.node_id, leaf_nodes_list)
  internal_nodes_id_list = map(lambda x: x.node_id, internal_nodes_list)
  print sorted(zip(leaf_nodes_id_list, counts[-27:,0]), key=lambda (x,y):x)

  for i, node in reversed(list(enumerate(internal_nodes_list))):
    for c in node.children:
      if c.isLeafNode:
        idx = num_internal_nodes + leaf_nodes_id_list.index(c.node_id)
      else:
        idx = internal_nodes_id_list.index(c.node_id)
      counts_cumulative[i] += counts_cumulative[idx]
  #print sorted(zip(leaf_nodes_id_list, counts_cumulative[-27:,0]), key=lambda (x,y):x)

  #print np.sum(counts_cumulative, axis=1)
  #print leaf_nodes_id_list
  #print internal_nodes_id_list

  counts_total = np.zeros(shape=(num_paths, num_classes))
  for i in range(num_paths):
    j = i + num_internal_nodes
    mult = 1.0
    while j >= 0:
      #print i, j
      counts_total[i] += mult * counts_cumulative[j, :]
      if j >= num_internal_nodes:  # leaf node
        try:
          j = internal_nodes_id_list.index(leaf_nodes_list[j - \
                                           num_internal_nodes].parent.node_id)
        except AttributeError:
          j = -1
      else:
        try:
          #print internal_nodes_list[j].node_id, internal_nodes_list[j].parent.node_id
          j = internal_nodes_id_list.index(internal_nodes_list[j].parent.node_id)
        except AttributeError:
          j = -1
      mult *= decay_factor
  #print counts_cumulative
  #print counts_total
  path2class = np.argmax(counts_total, axis=1)
  #print sorted(zip(leaf_nodes_id_list, path2class), key=lambda (x,y):x)
  #print path2class
  for i in range(len(path2class)):
    if np.max(counts_total[i]) == 0:
      #find closest path to this path
      dist = np.linalg.norm(alpha_values[i] - alpha_values, axis=1)
      min_idx = np.argmin(dist)
      while np.max(counts_total[min_idx]) == 0:
        dist[min_idx] = np.inf
        min_idx = np.argmin(dist)
      path2class[i] = path2class[min_idx]
  #print sorted(zip(leaf_nodes_id_list, path2class), key=lambda (x,y):x)
  z_test = np.loadtxt(z_test_file)
  test_labels = np.loadtxt(test_labels_file) - 1
  predicted_labels = []
  for z in z_test:
    closest_path_idx = np.argmin(np.linalg.norm((alpha_values - z[2:]), axis=1))
    predicted_labels.append(path2class[closest_path_idx])
  
  corr = map(lambda (x, y): 1 if (x==y) else 0, zip(predicted_labels, test_labels))
  predicted_labels = np.asarray(predicted_labels)
  aggr_corr = 0.0
  aggr_total = 0.0
  for class_id in range(15):
    data_pts_idx_r = np.where(test_labels == class_id)[0]
    corr_r = [corr[i] for i in data_pts_idx_r]
    recall = (100.0 * np.sum(corr_r)) / len(corr_r)
    data_pts_idx_p = np.where(predicted_labels == class_id)[0]
    corr_p = [corr[i] for i in data_pts_idx_p]
    precision = (100.0 * np.sum(corr_p)) / len(corr_p)
    aggr_corr += np.sum(corr_p)
    aggr_total += len(corr_p)
    #print precision
    print len(corr_r) / (1.0 * len(corr))
  print 100.0 * aggr_corr / aggr_total
  '''
  for class_id in range(15):
    data_pts_idx = np.where(test_labels==class_id)[0]
    corr_ = [corr[i] for i in data_pts_idx]
    print class_id, (100.0 * np.sum(corr_)) / len(corr_)
  print (np.sum(corr) * 100.0) / len(corr)
  '''

if __name__ == '__main__':
  nodes_file = sys.argv[1]
  train_labels_file = sys.argv[2]
  z_train_file = sys.argv[3]
  test_labels_file = sys.argv[4]
  z_test_file = sys.argv[5]
  decay_factor = eval(sys.argv[6])
  main(nodes_file, train_labels_file, z_train_file, test_labels_file, z_test_file, \
       decay_factor)
  '''
  np.set_printoptions(threshold=np.inf)
  alpha_file = sys.argv[1]
  sigma_file = sys.argv[2]
  z_train_file = sys.argv[3]
  z_test_file = sys.argv[4]
  bf = eval(sys.argv[5])
  num_levels = eval(sys.argv[6])
  num_paths = eval(sys.argv[7])
  train_labels_file = sys.argv[8]
  test_labels_file = sys.argv[9]
  decay_factor = eval(sys.argv[10])
  main(alpha_file, sigma_file, z_train_file, z_test_file, bf, num_levels, num_paths, \
       train_labels_file, test_labels_file, decay_factor)
  '''
