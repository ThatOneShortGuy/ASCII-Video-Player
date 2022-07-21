import numpy as np
from PIL import Image, ImageOps
import os

m = [(np.mean(ImageOps.grayscale(Image.open(f'ascii\\{i}'))), chr(int(i[:-4]))) for i in os.listdir('ascii')]


m.append((254.8, '█'))
m.append((131.5263, '■'))
m.append((194.899, '▓'))
m.append((67.499789, '▒'))
m.append((34.1162, '░'))
m.append((84.73, '╬'))
m.append((207.0305, '◘'))
m.append((100.588, '◙'))
m.append((79.2804, '¶'))

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

arr = [None] * 256
for i, e in m.items():
    arr[i] = e

with open('ascii_darkmap.dat', 'wb') as f:
    for e in arr:
        f.write(ord(e).to_bytes(2, 'big'))
