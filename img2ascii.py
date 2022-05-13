from PIL import Image, ImageOps
import sys
import json

SIZE = 128, 200


def load_ascii_map(filename):
    with open(filename) as f:
        m = json.load(f)
    return {int(k): v for k, v in m.items()}


def img2ascii(img, ascii_map):
    s = ''
    img.thumbnail(SIZE, Image.ANTIALIAS)
    h, w = img.size
    img = ImageOps.mirror(img)
    img = img.rotate(90, expand=1).load()
    for i in range(w):
        for j in range(h):
            s += ascii_map[img[i, j]]
        s += '\n'
    return s


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.argv.append('git.png')
    img = Image.open(sys.argv[1])
    img = img.convert('L')
    ascii_map = load_ascii_map('ascii_darkmap.json')
    s = img2ascii(img, ascii_map)
    print(s)
