import numpy as np
import scipy.stats
import sys
import copy
from node import Node
import cPickle
from PIL import Image

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

def write_image(output_filename, input_filenames, K):
  k1 = int(np.sqrt(K))
  new_im = Image.new('RGB', (k1 * 640, k1 * 360))
  for i in range(k1):
    for j in range(k1):
      curr_image = Image.open(input_filenames[i * k1 + j])
      if curr_image <> None:
        new_im.paste(curr_image.resize((636, 356)), (i * 640 + 2, j * 360 + 2))
  new_im.save(output_filename)

def get_node_to_imgs(node, result, z, K, filenames_list):
  if node.isLeafNode:
    frames_assigned = result[node]
  else:
    for c in node.children:
      result = get_node_to_imgs(c, result, z, K, filenames_list)
    frames_assigned = []
    for c in node.children:
      frames_assigned += list(result[c])
    result.update({node: frames_assigned})
  dist_to_path = np.linalg.norm(node.alpha - z[frames_assigned, 2:], axis=1)
  top_k = sorted(range(len(dist_to_path)), key=lambda i: dist_to_path[i])[:K]
  filenames_curr_node = map(lambda j: filenames_list[j].strip(), \
                            [frames_assigned[i] for i in top_k])
  try:
    print node.node_id, '\t', node.parent.node_id, '\t', filenames_curr_node
  except:
    print node.node_id, '\t', 'Root node', '\t', filenames_curr_node
  write_image('./output_imgs4/' + node.node_id + '.jpg', filenames_curr_node, K)
  return result

#def main(nodes_file, z_train_file, z_test_file, bf, num_levels, num_paths, \
#         train_labels_file, test_labels_file, decay_factor):
def main(nodes_filename, train_labels_file, z_train_file, test_labels_file, z_test_file, \
         decay_factor, filenames, K):
  filenames_list = open(filenames).readlines()
  nodes_file = open(nodes_filename, 'rb')
  nodes_list = cPickle.load(nodes_file)
  nodes_file.close()
  root_node = None
  for node in nodes_list:
    if node.parent == None:
      root_node = node
      break
  alpha_dict = get_alpha_leaves(root_node, {})
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
  
  leaf2imgs = {}
  for path_id in range(num_paths):
    frames_assigned = np.where(np.asarray(path_assignments) == path_id)[0]
    leaf2imgs.update({leaf_nodes_list[path_id] : \
                      list(frames_assigned)})
  nodes2imgs = get_node_to_imgs(root_node, leaf2imgs, z_train, K, filenames_list)

if __name__ == '__main__':
  nodes_file = sys.argv[1]
  train_labels_file = sys.argv[2]
  z_train_file = sys.argv[3]
  test_labels_file = sys.argv[2]
  z_test_file = sys.argv[3]
  decay_factor = 0.0
  filenames = sys.argv[4]
  K = eval(sys.argv[5])
  main(nodes_file, train_labels_file, z_train_file, test_labels_file, z_test_file, \
       decay_factor, filenames, K)
