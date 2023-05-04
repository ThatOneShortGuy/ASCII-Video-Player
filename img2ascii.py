import cv2
import sys
import os
from math import ceil, log, exp
from cimg2ascii import cmap_color, cmap_color_old, cget_color_samples, cinsert_color
import numpy as np

os.system("")

SIZE = 266, 61
COLOR_SAMPLE_FREQ = 100

def load_ascii_map(filename):
    with open(filename, 'rb') as f:
        arr = [chr(int.from_bytes(f.read(2), 'big')) for _ in range(256)]
    return arr


def img2ascii(img, ascii_map):
    w, h = img.shape
    return ''.join(''.join(map(lambda x: ascii_map[x], img[i])) + '\n' for i in range(w))

def estimate_insert_color_size(new_threshold, prev_threshold, prev_size, min_length):
    return min_length * exp(1.96 * exp(new_threshold * log(log(prev_size / min_length) / 1.96) / prev_threshold))

def get_optimal_threshold(old_threshold, old_size, size_at_old_thresh, max_length=2**14-1000):
    if size_at_old_thresh == old_size:
        return old_threshold + (50 - old_threshold) // 2
    return abs(int(old_threshold * log(log(max_length / old_size) / 1.96) / log(log(size_at_old_thresh / old_size) / 1.96)+1))

def predict_insert_color_size(img, threshold_of_change=10):
    h, w, _ = img.shape
    r_old = g_old = b_old = 255
    length = 0
    for col in range(h):
        for row in range(w):
            b, g, r = img[col, row]
            if abs(r - r_old) > threshold_of_change or abs(g - g_old) > threshold_of_change or abs(b - b_old) > threshold_of_change:
                str_to_add = f'\033[38;2;{r};{g};{b}m_'
                length += len(str_to_add)
                r_old, g_old, b_old = map(int, (r, g, b))
            else:
                length += 1
        length += 1
    return length

def insert_color(s, img, threshold_of_change=10):
    buffer = np.empty((len(s)*19,), dtype=np.unicode_)
    h, w, _ = img.shape
    r_old = g_old = b_old = 255
    length = 0
    for col in range(h):
        for row in range(w):
            b, g, r = img[col, row]
            char = s[col*(w+1) + row]
            if abs(r - r_old) > threshold_of_change or abs(g - g_old) > threshold_of_change or abs(b - b_old) > threshold_of_change:
                str_to_add = f'\033[38;2;{r};{g};{b}m{char}'
                for char in str_to_add:
                    buffer[length] = char
                    length += 1
                r_old, g_old, b_old = map(int, (r, g, b))
            else:
                buffer[length] = char
                length += 1
        buffer[length] = '\n'
        length += 1
    
    return ''.join(buffer[:length])


def get_color_samples(img, output, freq=COLOR_SAMPLE_FREQ):
    h, w, _ = img.shape
    color_locs = tuple(tuple(i+j for j in range(freq) if i+j < w) for i in range(0, w, freq))
    for i in range(h):
        for j in range(ceil(w/freq)):
            col = img[i, color_locs[j]].mean(axis=0).astype('uint8').tolist()
            try:
                output[i * ceil(w/freq) + j] = col
            except Exception as e:
                print(f'{i = } {j = } {w = } {freq = }')
                raise e
    # color_locs = tuple(i for i in range(0, w, freq))
    # return img[:, color_locs].reshape((h*len(color_locs),3)).tolist()

def get_colored_ascii(img, ascii_map, freq=COLOR_SAMPLE_FREQ, color=True):
    w, h, _ = img.shape
    output = [None]*((ceil(h/freq))*w)
    get_color_samples(img, output, freq)
    # coutput = cget_color_samples(img, freq)
    
    # print(output)
    no_color_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = img2ascii(no_color_img, ascii_map)
    # s = '\n'.join('█' * h for _ in range(w)) + '\n'
    # old = map_color(s, output, h+1, freq)
    new = cmap_color(s, img, h+1, freq, 0)
    return new if color else no_color_img

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


def calc_ratio(w, h, img):
    iw, ih = img.shape[:2]
    if w == -1:
        w = int(iw * h / ih)
    elif h == -1:
        h = int(ih * w / iw)
    return w, h

def main():
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
            w, h = map(int, sys.argv.pop(1).split(':'))
        else:
            print('''\nUsage: img2ascii.py <input_file> [options]\n
    Options:
        -f <freq>, -c <freq> : Color sample frequency. Can't be lower than 1 (default: 2),
        -s <width>:<height> : Size of the output image (default: 266:61),
        -h : Show this help message''')
            sys.exit(0)

    img = cv2.imread(input_file)
    w, h = calc_ratio(w, h, img)
    img = cv2.resize(img, (w, h))
    freq = max(1, freq)
    ascii_map = load_ascii_map('ascii_darkmap.dat')
    # ns = get_colored_ascii(img, ascii_map, freq)
    # print(ns)
    # print(f"Length of old string: {len(ns)}")
    no_color_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = img2ascii(no_color_img, ascii_map)
    ns = cinsert_color(s, img, freq)
    print(ns)
    
    return ns

if __name__ == "__main__":
    ns = main()
