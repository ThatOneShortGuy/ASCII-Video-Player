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

os.add_dll_directory(os.getcwd())

SIZE = 188, 40
COLOR_SAMPLE_FREQ = 16

def get_vid_dim(path):
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h

def read_video(video_path, fps=None, freq=COLOR_SAMPLE_FREQ, size=SIZE):
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
            os.system(f'ffmpeg -y -i "{video_path}" -vn temp/out.flac')
            os.system(f'ffmpeg -y -i "{video_path}" -pix_fmt rgb24 -vf fps={fps},scale={size[0]}:{size[1]} temp/%08d.png')

        try:
            img = 1
            p = vlc.MediaPlayer('temp/out.flac')
            while not os.path.exists(f'temp/{img:08d}.png'):
                pass
            p.play()
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


def show_video(path, fps, freq=COLOR_SAMPLE_FREQ, size=SIZE):
    ifps = 1 / fps
    ascii_map = load_ascii_map('ascii_darkmap.dat')
    start = perf_counter()
    for frame in read_video(path, fps, freq=freq, size=size):
        if frame is None:
            break
        colors = cget_color_samples(frame, freq)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        s = img2ascii(frame, ascii_map)
        # s = '\n'.join('█' * frame.shape[1] for _ in range(frame.shape[0])) + '\n'
        s = cmap_color(s, colors, frame.shape[1]+1, freq, 1)
        # print(f'S: {len(s)}, freq: {freq}')
        if len(s) > 15500:
            freq += 1
        elif len(s) < 13000:
            freq -= 1
        freq = max(1, freq)
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
        --clean : Clean the temporary files before and after the program is done running (default: False)
        -f <freq>, -c <freq> : Color sample frequency. Can't be lower than 1 (default: 16)
        -fps <fps>, -r <fps> : Framerate of the output video (default: 30)
        -s <width>,<height> : Size of the output video (default: 188,40)
        -h : Show this help message'''
    video_path = 'crf18.mp4'
    clean = False
    freq = COLOR_SAMPLE_FREQ
    w, h = SIZE
    fps = 30
    if len(sys.argv) < 2 and not os.path.isfile(video_path):
        print(usage)
        sys.exit(0)
    video_path = sys.argv.pop(1) if len(sys.argv) > 1 else video_path
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
        elif val in ('--clean'):
            if os.path.exists('temp'):
                shutil.rmtree('temp')
            clean = True
        else:
            print(usage)
            sys.exit(0)

    try:
        show_video(video_path, fps, freq=freq, size=(w, h))
    finally:
        # if clean:
        #     shutil.rmtree('temp')
        print('\033[0m')
