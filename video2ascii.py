from img2ascii import img2ascii, load_ascii_map, get_color_samples, calc_ratio
from random import randint
from cimg2ascii import cmap_color
import sys
import os
import cv2
from time import perf_counter
from threading import Thread
import shutil
import vlc

os.add_dll_directory(os.getcwd())

SIZE = 188, 40
COLOR_SAMPLE_FREQ = 22

def read_video(video_path, fps=None, freq=COLOR_SAMPLE_FREQ, size=SIZE):
    """
    Reads a video file and returns a list of frames using ffmpeg
    """
    try:
        if not os.path.exists('temp'):
            os.mkdir('temp')
            os.system(f'ffmpeg -y -i "{video_path}" -vn temp/out.flac')
            os.system(
                f'ffmpeg -y -i "{video_path}" -pix_fmt rgb24 {f"-vf fps={fps},scale={size[0]}:{size[1]} " if fps else ""}temp/%08d.png')

        try:
            img = 1
            p = vlc.MediaPlayer('temp/out.flac')
            while not os.path.exists(f'temp/{img:08d}.png'):
                pass
            p.play()
            while True:
                read_img = cv2.imread(f'temp/{img:08d}.png')
                colors = get_color_samples(read_img, freq)
                yield cv2.cvtColor(read_img, cv2.COLOR_BGR2GRAY), colors
                # os.remove(f'temp/{img:08d}.jpg')
                img += 1
        except Exception:
            pass
        finally:
            p.release()
    finally:
        pass
        # shutil.rmtree('temp')


def show_video(path, fps, freq=COLOR_SAMPLE_FREQ, size=SIZE):
    ifps = 1 / fps
    ascii_map = load_ascii_map('ascii_darkmap.dat')
    start = perf_counter()
    for frame, colors in read_video(path, fps, freq=freq, size=size):

        s = img2ascii(frame, ascii_map)
        # s = '\n'.join('█' * frame.shape[1] for _ in range(frame.shape[0])) + '\n'
        global COLOR_SAMPLE_FREQ
        s = cmap_color(s, colors, frame.shape[1], COLOR_SAMPLE_FREQ)
        # COLOR_SAMPLE_FREQ = 34
        # COLOR_SAMPLE_FREQ = randint(22, 30)
        thread = Thread(target=sys.stdout.write, args=(s,))
        while ((end := perf_counter()) - start) < ifps:
            pass
        start = end
        # st = perf_counter()
        thread.start()
        # print(perf_counter() - st)
        # break


if __name__ == "__main__":
    usage = '''\nUsage: vid2ascii.py <input_file> [options]\n
    Options:
        -f <freq>, -c <freq> : Color sample frequency. Can't be lower than 1 (default: 22),
        -fps <fps>, -r <fps> : Framerate of the output video (default: 30),
        -s <width>,<height> : Size of the output video (default: 188,40),
        -h : Show this help message'''
    video_path = 'crf18.pm4'
    freq = COLOR_SAMPLE_FREQ
    w, h = SIZE
    fps = 30
    if len(sys.argv) < 2:
        print(usage)
    video_path = sys.argv.pop(1)
    if not os.path.isfile(video_path):
        print(f'File "{video_path}" not found')
        sys.exit(1)
    while len(sys.argv) > 1:
        val = sys.argv.pop(1)
        if val in ('-f', '-c'):
            freq = int(sys.argv.pop(1))
        elif val in ('-s'):
            w, h = map(int, sys.argv.pop(1).split(','))
        elif val in ('-fps', '-r'):
            fps = int(sys.argv.pop(1))
        else:
            print(usage)
            sys.exit(0)

    try:
        show_video(video_path, fps, freq=freq, size=(w, h))
    finally:
        print('\033[0m')
