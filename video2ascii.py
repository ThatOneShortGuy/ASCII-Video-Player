from img2ascii import img2ascii, load_ascii_map, get_color_samples, map_color, insert_color, predict_insert_color_size, get_optimal_threshold
from cimg2ascii import cmap_color, cget_color_samples, cpredict_insert_color_size, cinsert_color
import sys
import os
import cv2
import time
import subprocess
import numpy as np

SIZE = 170, -1
COLOR_SAMPLE_FREQ = 10

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

def process_ffmpeg_args(args):
    vfilters = ''
    afilters = ''
    extras = []
    while len(args) > 0:
        val = args.pop(0)
        if val in ('-vf', '-filter:v'):
            vfilters = args.pop(0)
        elif val in ('-af', '-filter:a'):
            afilters = args.pop(0)
        else:
            extras.append(val)
    return vfilters, afilters, ' '.join(extras)


def read_video(video_path, fps=None, freq=COLOR_SAMPLE_FREQ, size=SIZE, start_time=None, ffmpeg=[]):
    """
    Reads a video file and returns a list of frames using ffmpeg
    """
    if size[0] == -1:
        w, h = get_vid_dim(video_path)
        size = int(w * size[1] / h *2), size[1]
    elif size[1] == -1:
        w, h = get_vid_dim(video_path)
        size = size[0], int(h * size[0] / w / 2)

    vfilters, afilters, ffmpeg = process_ffmpeg_args(ffmpeg)

    vidp = subprocess.Popen(
        f'ffmpeg -i "{video_path}" -pix_fmt bgr24 -vf fps={fps},{vfilters+"," if vfilters else ""}scale={size[0]}:{size[1]} {ffmpeg} -f rawvideo -',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL)
    vida = subprocess.Popen(
        f'ffmpeg -i "{video_path}" -vn -f wav {f"-af {afilters}" if afilters else ""} {ffmpeg} -',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL)
    
    print('Loading video...')
    data = vidp.stdout.read(3*size[0]*size[1])
    data = np.frombuffer(data, dtype='uint8').reshape((size[1], size[0], 3))
    
    try:
        p = subprocess.Popen('ffplay -nodisp -autoexit -loglevel error -stats -i pipe:0 -f wav', shell=True, stdin=vida.stdout, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL)
        audio=True
    except FileNotFoundError:
        print("\n\033[31mCould not locate installation for FFplay.\nPlaying video without audio\033[0m")
        time.sleep(2)
        audio=False
    try:
        # data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        p.stderr.read(69)
        yield data
        if start_time is not None:
            start_time[0] = time.time()
        while data:=vidp.stdout.read(3*size[0]*size[1]):
            data = np.frombuffer(data, dtype='uint8').reshape((size[1], size[0], 3))
            # data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            yield data
            # os.remove(f'temp/{img:08d}.jpg')
    finally:
        vidp.stdout.close()
        vidp.kill()
        vida.stdout.close()
        vida.kill()
        if audio:
            p.kill()


def show_video(path, fps, freq=COLOR_SAMPLE_FREQ, size=SIZE, ffmpeg='', debug=False, no_ascii=False, colorless=False):
    fps = get_vid_fps(video_path) if fps is None else fps
    path_to_self = os.path.dirname(os.path.realpath(__file__))
    ascii_map = load_ascii_map(os.path.join(path_to_self,'ascii_darkmap.dat'))
    start = [time.time()]
    for i, frame in enumerate(read_video(path, fps, freq=freq, size=size, start_time=start, ffmpeg=ffmpeg)):
        if frame is None:
            break
        # colors = cget_color_samples(frame, freq)
        colorless_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        s = '\n'.join('█' * frame.shape[1] for _ in range(frame.shape[0])) + '\n' if no_ascii else img2ascii(colorless_frame, ascii_map)
        # s = cmap_color(s, frame, frame.shape[1]+1, freq, 1) if not colorless else s
        while freq > 1 and cpredict_insert_color_size(frame, freq) < 16000:
            freq -= 5
        freq = max(freq, 1)
        while cpredict_insert_color_size(frame, freq) > 16200 and freq < 255:
            freq += 1
        ns = cinsert_color(s, frame, freq) if not colorless else s
        
        while i > fps * (time.time() - start[0]):
            pass
        sys.stdout.write(f'\033[0m\nfreq: {freq}\tframe: {i}\tfps: {i/(time.time()-start[0]):.4g} \tstrlen: {len(ns)}\n'+ns
                         if debug else ns)

if __name__ == "__main__":
    video_path = 'crf18.mp4'
    colorless = False
    debug = False
    freq = COLOR_SAMPLE_FREQ
    no_ascii = False
    w, h = SIZE
    fps = None
    ffmpeg = ''
    # Add args if python run in debug mode
    if sys.gettrace():
        sys.argv.extend(['D:\RIFE app\Waifu Racks\out.mkv','-d', '-s', '10:-1'])
    usage = f'''\nUsage: vid2ascii.py <input_file> [options]\n
    Options:
        -h : Show this help message

        --colorless : Don't use color in the output (default: {colorless})
        -d, --debug : Show debug information (default: {debug})
        -f <freq>, -c <freq> : Color sample frequency. Can't be lower than 1 or greater than the width (default: {freq})
        --no-ascii : Don't use ascii characters to represent the video (default: {no_ascii})
        -r <fps>, --fps <fps> : Framerate of the output video (default: video's framerate)
        -s <width>:<height> : Size of the output video (default: {SIZE[0]}:{SIZE[1]})
        
        --ffmpeg [...] : All commands after this will be passed to ffmpeg'''
    if len(sys.argv) < 2 and not os.path.isfile(video_path):
        print(usage)
        sys.exit(0)
    video_path = sys.argv.pop(1) if len(sys.argv) > 1 else video_path
    if not os.path.isfile(video_path):
        print(f'File "{video_path}" not found')
        sys.exit(1)
    while len(sys.argv) > 1:
        val = sys.argv.pop(1)
        if val in ('--colorless'):
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
        elif val in ('--ffmpeg'):
            ffmpeg = sys.argv[1:]
            break
        else:
            print(usage)
            sys.exit(0)

    try:
        show_video(video_path, fps=fps, freq=freq, size=(w, h), ffmpeg=ffmpeg, debug=debug, no_ascii=no_ascii, colorless=colorless)
    except KeyboardInterrupt:
        pass
    finally:
        print('\033[0m')
