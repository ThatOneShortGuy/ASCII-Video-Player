from PIL import Image
import sys
import numpy as np


SIZE = 130, 65


def load_ascii_map(filename):
    with open(filename, 'rb') as f:
        arr = [chr(int.from_bytes(f.read(2), 'big')) for _ in range(256)]
    return np.array(arr)


def img2ascii(img, ascii_map, size=SIZE):
    img = img.resize(size)
    h, w = img.size
    img = np.array(img)
    return ''.join(''.join(map(lambda x: ascii_map[x], img[i])) + '\n' for i in range(w))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.argv.append('git.png')
    img = Image.open(sys.argv[1])
    img = img.convert('L')
    ascii_map = load_ascii_map('ascii_darkmap.dat')
    s = img2ascii(img, ascii_map)
    print(s)
