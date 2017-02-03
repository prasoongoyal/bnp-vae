import numpy as np
from PIL import Image

data = np.load('/work/ans556/prasoon/Features/full_data_th.npy')
labels = open('/work/ans556/prasoon/train_list.txt').readlines()

for i, l in enumerate(labels):
  l = l.strip()
  l = l.split('/')[-1]
  im = Image.fromarray(np.reshape(data[i, :100] * 255.0, (10, 10)))
  im = im.convert('RGB')
  im.save('/scratch/ans556/prasoon/images/' + l, 'JPEG')
