from img2ascii import img2ascii, load_ascii_map, get_color_samples, map_color
from cimg2ascii import cmap_color, cget_color_samples
import sys
import os
from math import ceil
import cv2
from time import perf_counter
from threading import Thread
import shutil
import vlc
import time

os.add_dll_directory(os.getcwd())

SIZE = 188, 40
COLOR_SAMPLE_FREQ = 16

def get_vid_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def get_vid_dim(path):
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h

def read_video(video_path, fps=None, freq=COLOR_SAMPLE_FREQ, size=SIZE, start_time=None, ffmpeg=''):
    """
    Reads a video file and returns a list of frames using ffmpeg
    """
    if size[0] == -1:
        w, h = get_vid_dim(video_path)
        size = int(w * size[1] / h *2), size[1]
    elif size[1] == -1:
        w, h = get_vid_dim(video_path)
        size = size[0], int(h * size[0] / w / 2)

    try:
        if not os.path.exists('temp'):
            os.mkdir('temp')
            os.system(f'ffmpeg -y -i "{video_path}" -vn -c:a copy temp/audio.mkv')
            os.system(f'ffmpeg -y -i "{video_path}" -pix_fmt rgb24 -vf fps={fps},scale={size[0]}:{size[1]} {ffmpeg} temp/%08d.png')

        try:
            img = 1
            p = vlc.MediaPlayer('temp/audio.mkv')
            while not os.path.exists(f'temp/{img:08d}.png'):
                pass
            p.play()
            if start_time is not None:
                start_time[0] = time.time()
            while True:
                read_img = cv2.imread(f'temp/{img:08d}.png')
                yield read_img
                # os.remove(f'temp/{img:08d}.jpg')
                img += 1
        except Exception:
            pass
        finally:
            p.release()
    finally:
        pass


def show_video(path, fps, freq=COLOR_SAMPLE_FREQ, size=SIZE, ffmpeg='', debug=False, no_ascii=False, colorless=False):
    fps = get_vid_fps(video_path) if fps is None else fps
    ascii_map = load_ascii_map('ascii_darkmap.dat')
    start = [time.time()]
    for i, frame in enumerate(read_video(path, fps, freq=freq, size=size, start_time=start, ffmpeg=ffmpeg)):
        if frame is None:
            break
        # colors = cget_color_samples(frame, freq)
        colorless_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        s = '\n'.join('█' * frame.shape[1] for _ in range(frame.shape[0])) + '\n' if no_ascii else img2ascii(colorless_frame, ascii_map)
        s = cmap_color(s, frame, frame.shape[1]+1, freq, 1) if not colorless else s
        # print(f'S: {len(s)}, freq: {freq}')
        if len(s) > 15500:
            freq += 1
            freq = min(frame.shape[1], freq)
        elif len(s) < 14500:
            freq -= 1
            freq = max(1, freq)
        while i > fps * (time.time() - start[0]):
            pass
        sys.stdout.write(f'\033[0m\nfreq: {freq}, frame: {i}, fps: {fps:.4g}, strlen: {len(s)}\n'+s if debug else s)


if __name__ == "__main__":
    usage = '''\nUsage: vid2ascii.py <input_file> [options]\n
    Options:
        -h : Show this help message

        --clean : Clean the temporary files before running (default: False)
        --colorless : Don't use color in the output (default: False)
        -d, --debug : Show debug information (default: False)
        -f <freq>, -c <freq> : Color sample frequency. Can't be lower than 1 or greater than the width (default: 16)
        --no-ascii : Don't use ascii characters to represent the video (default: False)
        -r <fps>, --fps <fps> : Framerate of the output video (default: video's framerate)
        -s <width>:<height> : Size of the output video (default: 188:40)
        
        --ffmpeg [...] : All commands after this will be passed to ffmpeg video'''
    video_path = 'crf18.mp4'
    clean = False
    colorless = False
    debug = False
    freq = COLOR_SAMPLE_FREQ
    no_ascii = False
    w, h = SIZE
    fps = None
    ffmpeg = ''
    if len(sys.argv) < 2 and not os.path.isfile(video_path):
        print(usage)
        sys.exit(0)
    video_path = sys.argv.pop(1) if len(sys.argv) > 1 else video_path
    if not os.path.isfile(video_path):
        print(f'File "{video_path}" not found')
        sys.exit(1)
    while len(sys.argv) > 1:
        val = sys.argv.pop(1)
        if val in ('--clean'):
            if os.path.exists('temp'):
                shutil.rmtree('temp')
        elif val in ('--colorless'):
            colorless = True
        elif val in ('-d', '--debug'):
            debug = True
        elif val in ('-f', '-c'):
            freq = int(sys.argv.pop(1))
        elif val in ('--no-ascii'):
            no_ascii = True
        elif val in ('-r', '--fps'):
            fps = int(sys.argv.pop(1))
        elif val in ('-s'):
            w, h = map(int, sys.argv.pop(1).split(':'))
            clean = True
        elif val in ('--ffmpeg'):
            ffmpeg = ' '.join(sys.argv[1:])
            break
        else:
            print(usage)
            sys.exit(0)

    try:
        show_video(video_path, fps=fps, freq=freq, size=(w, h), ffmpeg=ffmpeg, debug=debug, no_ascii=no_ascii, colorless=colorless)
    finally:
        # if clean:
        #     shutil.rmtree('temp')
        print('\033[0m')
