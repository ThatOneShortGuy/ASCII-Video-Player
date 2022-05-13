import numpy as np
from PIL import Image, ImageOps
import os
import json

m = []

for i in os.listdir('ascii'):
    m.append((np.mean(ImageOps.grayscale(Image.open(f'ascii\\{i}'))), chr(int(i[:-4]))))

arr, char = zip(*m)
arr = np.array(arr)
arr *= 255/np.max(arr)
arr = np.array(arr, dtype=np.uint8)
arr = dict(zip(arr, char))

keys = arr.keys()
m = arr.copy()
for i in range(256):
    if i not in keys:
        for j in range(1, 256):
            if i + j in keys:
                m[i] = arr[i+j]
                break
            if i - j in keys:
                m[i] = arr[i-j]
                break

m = {str(k): v for k, v in m.items()}

with open('ascii_darkmap.json', 'w') as f:
    json.dump(m, f)
