import numpy as np
import sys

def main(z_filename, alpha_filename):
  z = np.loadtxt(z_filename)
  a = np.loadtxt(alpha_filename)
  for i in range(len(z)):
    dist = np.linalg.norm(a - z[i, 2:], axis=1)
    #top10 = sorted(range(len(dist)), key=lambda x: dist[x])[:10]
    #print i, top10
    print int(z[i, 0]), int(z[i, 1]), np.argmin(dist)
  
if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2])
