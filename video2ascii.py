from img2ascii import img2ascii, load_ascii_map
import sys
import os
from PIL import Image
from time import perf_counter, sleep
import threading
import shutil

SIZE = 236, 60


def read_video(video_path):
    """
    Reads a video file and returns a list of frames using ffmpeg
    """
    if not os.path.exists('temp'):
        os.mkdir('temp')
        threading.Thread(target=os.system, args=(
            f'ffmpeg -hide_banner -loglevel error -i {video_path} -q:v 31 temp/%08d.jpg',)).start()
        sleep(.05)
    try:
        img = 1
        while not os.path.exists(f'temp/{img:08d}.jpg'):
            pass
        while True:
            yield Image.open(f'temp/{img:08d}.jpg')
            os.remove(f'temp/{img:08d}.jpg')
            img += 1
    except FileNotFoundError:
        pass
    except Exception as e:
        print(e)
    finally:
        shutil.rmtree('temp')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.argv.append('clip.mkv')
    video_path = sys.argv[1]
    ascii_map = load_ascii_map('ascii_darkmap.json')
    fps = 100
    times = []
    start = perf_counter()
    for frame in read_video(video_path):
        frame = frame.convert('L')
        s = img2ascii(frame, ascii_map, SIZE)
        while (end := perf_counter()) - start < (1 / (fps)):
            pass
        start = end
        print(s)
