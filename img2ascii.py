import cv2
import sys
import os
from cimg2ascii import cinsert_color

os.system("")

SIZE = 266, -1
COLOR_SAMPLE_FREQ = 1

class Args:
    img_path = 'git.png'
    colorless = False
    freq = COLOR_SAMPLE_FREQ
    no_ascii = False
    w, h = SIZE

def load_ascii_map(filename):
    with open(filename, 'rb') as f:
        arr = [chr(int.from_bytes(f.read(2), 'big')) for _ in range(256)]
    return arr


def img2ascii(img, ascii_map):
    w, h = img.shape
    return ''.join(''.join(map(lambda x: ascii_map[x], img[i])) + '\n' for i in range(w))

def calc_ratio(w, h, img):
    ih, iw = img.shape[:2]
    if w == -1:
        w = int(iw * h / ih)
    elif h == -1:
        h = int(ih * w / iw)
    return w, h

def main():
    help_message = '''\nUsage: img2ascii.py <input_file> [options]\n
    Options:
        -f <freq>, -c <freq> : Color sample frequency. Can't be lower than 1 (default: 2),
        -s <width>:<height> : Size of the output image (default: 266:61),
        --no-ascii : Don't use ascii characters (default: False),
        --no-color : Don't use colors (default: False),
        -h : Show this help message'''
    input_file = sys.argv.pop(1) if len(sys.argv) > 1 else Args.img_path
    if input_file == '-h':
        print(help_message)
        sys.exit(0)
    if not os.path.isfile(input_file):
        print(f'File "{input_file}" not found')
        sys.exit(1)
    while len(sys.argv) > 1:
        val = sys.argv.pop(1)
        if val in ('-f', '-c'):
            Args.freq = int(sys.argv.pop(1))
        elif val in ('-s'):
            Args.w, Args.h = map(int, sys.argv.pop(1).split(':'))
        elif val == '--no-ascii':
            Args.no_ascii = True
        elif val == '--no-color':
            Args.colorless = True
        else:
            print(help_message)
            sys.exit(0)

    img = cv2.imread(input_file)
    w, h = calc_ratio(Args.w, Args.h, img)
    img = cv2.resize(img, (w, h))
    freq = max(1, Args.freq)
    path_to_self = os.path.dirname(os.path.realpath(__file__))
    ascii_map = load_ascii_map(os.path.join(path_to_self,'ascii_darkmap.dat'))
    # ns = get_colored_ascii(img, ascii_map, freq)
    # print(ns)
    # print(f"Length of old string: {len(ns)}")
    no_color_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = '\n'.join('█' * w for _ in range(h)) + '\n' if Args.no_ascii else img2ascii(no_color_img, ascii_map)
    ns = s if Args.colorless else cinsert_color(s, img, freq)
    print(ns)
    print('\033[0m')
    return ns

if __name__ == "__main__":
    ns = main()
