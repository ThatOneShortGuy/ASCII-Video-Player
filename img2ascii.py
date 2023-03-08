import cv2
import sys
import os
from cimg2ascii import cmap_color, cmap_color_old

os.system("")

SIZE = 266,61
COLOR_SAMPLE_FREQ = 2

def load_ascii_map(filename):
    with open(filename, 'rb') as f:
        arr = [chr(int.from_bytes(f.read(2), 'big')) for _ in range(256)]
    return arr


def img2ascii(img, ascii_map):
    w, h = img.shape
    return ''.join(''.join(map(lambda x: ascii_map[x], img[i])) + '\n' for i in range(w))

def get_color_samples(img, freq=COLOR_SAMPLE_FREQ, offset=0):
    w, h, _ = img.shape
    color_locs = tuple(i for i in range(offset % freq, h, freq))
    return img[:, color_locs].reshape((w*len(color_locs),3)).tolist()
        
    # return [tuple(img[divmod(i, h)]) for i in range(freq, w * h, freq)]

def map_color_old(s, colors, h, freq=COLOR_SAMPLE_FREQ):
    ns = ''
    j = 0
    for i in range(len(s)):
        if j >= freq and not j % freq:
            # print((j-1) // COLOR_SAMPLE_FREQ)
            b, g, r = colors[(j-1) // freq]
            ns += f'\033[38;2;{r};{g};{b}m{s[i]}\033[0m' # Maximum of 24 characters
        else:
            ns += s[i]
        j += (i % h)!= 0
    return ns

def map_color(s, colors, line_len, freq=COLOR_SAMPLE_FREQ):
    ns = ''
    j = 0
    for i in range(len(s)):
        if s[i] == '\n' or (i % line_len) % freq:
            ns += s[i]
            continue
        b, g, r = colors[j]
        ns += f'\033[38;2;{r};{g};{b}m{s[i]}'
        j += 1
    return ns

def get_colored_ascii(img, ascii_map, freq=COLOR_SAMPLE_FREQ):
    colors = get_color_samples(img, freq)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w, h = img.shape
    s = img2ascii(img, ascii_map)
    # s = '\n'.join('█' * h for _ in range(w)) + '\n'
    old = map_color(s, colors, h+1, freq)
    new = cmap_color(s, colors, h+1, freq, 0)
    return new

def calc_ratio(w, h, img):
    iw, ih = img.shape[:2]
    if w == -1:
        w = int(iw * h / ih)
    elif h == -1:
        h = int(ih * w / iw)
    return w, h

if __name__ == "__main__":
    input_file = 'git.png'
    freq = COLOR_SAMPLE_FREQ
    w, h = SIZE
    input_file = sys.argv.pop(1) if len(sys.argv) > 1 else input_file
    if not os.path.isfile(input_file):
        print(f'File "{input_file}" not found')
        sys.exit(1)
    while len(sys.argv) > 1:
        val = sys.argv.pop(1)
        if val in ('-f', '-c'):
            freq = int(sys.argv.pop(1))
        elif val in ('-s'):
            w, h = map(int, sys.argv.pop(1).split(','))
        else:
            print('''\nUsage: img2ascii.py <input_file> [options]\n
    Options:
        -f <freq>, -c <freq> : Color sample frequency. Can't be lower than 1 (default: 2),
        -s <width>,<height> : Size of the output image (default: 266,61),
        -h : Show this help message''')
            sys.exit(0)

    img = cv2.imread(input_file)
    w, h = calc_ratio(w, h, img)
    img = cv2.resize(img, (w, h))
    freq = max(1, freq)
    ascii_map = load_ascii_map('ascii_darkmap.dat')
    ns = get_colored_ascii(img, ascii_map, freq)
    print(ns)
