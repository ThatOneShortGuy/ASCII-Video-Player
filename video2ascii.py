from img2ascii import img2ascii, load_ascii_map
import sys
import os
from PIL import Image
from time import perf_counter
import shutil
import vlc
import time


SIZE = 236, 63


def read_video(video_path, fps=None):
    """
    Reads a video file and returns a list of frames using ffmpeg
    """
    try:
        if not os.path.exists('temp'):
            os.mkdir('temp')
            os.system(f'ffmpeg -y -i "{video_path}" -vn temp/out.flac')
            os.system(f'ffmpeg -y -i "{video_path}" -q:v 31 {f"-vf fps={fps} " if fps else ""}temp/%08d.jpg')

        try:
            img = 1
            p = vlc.MediaPlayer('temp/out.flac')
            while not os.path.exists(f'temp/{img:08d}.jpg'):
                pass
            p.play()
            while True:
                yield Image.open(f'temp/{img:08d}.jpg')
                os.remove(f'temp/{img:08d}.jpg')
                img += 1
        except FileNotFoundError:
            pass
        except Exception as e:
            print(e)
        finally:
            p.stop()
            p.release()
    finally:
        shutil.rmtree('temp')


def show_video(path, fps=55):
    ascii_map = load_ascii_map('ascii_darkmap.json')
    start = perf_counter()
    for frame in read_video(video_path, fps):
        frame = frame.convert('L')
        s = img2ascii(frame, ascii_map, SIZE)
        while (end := perf_counter()) - start < (1 / (fps)):
            pass
        start = end
        # print "s" to stdout the fast way
        sys.stdout.write(s)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.argv.append('crf18.mp4')
        sys.argv.append('60')
    video_path = sys.argv[1]
    fps = float(sys.argv[2])
    show_video(video_path, fps)
