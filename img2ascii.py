import cv2
import sys


SIZE = 250, 53


def load_ascii_map(filename):
    with open(filename, 'rb') as f:
        arr = [chr(int.from_bytes(f.read(2), 'big')) for _ in range(256)]
    return arr


def img2ascii(img, ascii_map, size=SIZE):
    # img = cv2.resize(img, size, cv2.INTER_CUBIC)
    w, h = img.shape
    return ''.join(''.join(map(lambda x: ascii_map[x], img[i])) + '\n' for i in range(w))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.argv.append('git.png')
    img = cv2.imread(sys.argv[1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ascii_map = load_ascii_map('ascii_darkmap.dat')
    s = img2ascii(img, ascii_map)
    print(s)
