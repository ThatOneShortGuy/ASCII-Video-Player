import io
import os
import subprocess
import sys
import time
import numpy as np

from cimg2ascii import cinsert_color, cimg2ascii, cget_freq
from img2ascii import load_ascii_map

SIZE = 250, -1
MAX_CHARS = 32500

print(io.DEFAULT_BUFFER_SIZE)

class Args:
    video_path = 'crf18.mp4'
    colorless = False
    debug = False
    freq = 30
    interlace = 1
    no_ascii = False
    max_chars = MAX_CHARS
    min_freq = 10
    size = SIZE
    start_time = 0
    fps = None
    tempo = 1
    ffmpeg = ''

def get_vid_fps(path):
    p = subprocess.Popen(f'ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate -i "{path}"',
                           shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fps = p.stdout.read().decode().split('/')
    fps = int(fps[0])/int(fps[1])
    p.stdout.close()
    p.terminate()
    return fps

def get_vid_dim(path):
    p = subprocess.Popen(f'ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=width,height -i "{path}"',
                           shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    w, h = p.stdout.read().decode().split(',')
    p.stdout.close()
    p.terminate()
    return int(w), int(h)

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

def get_video_size(args: Args):
    terminal_size = os.get_terminal_size()
    w, h = get_vid_dim(args.video_path)

    if args.size[0] == -1:
        size = int(w * args.size[1] / h *2), args.size[1]
    elif args.size[1] == -1:
        size = args.size[0], int(h * args.size[0] / w / 2)
    else:
        size = args.size
    
    args.size = size

    if size[0] > terminal_size[0]:
        args.size = terminal_size[0]-2, -1
        args.size = get_video_size(args)
    elif size[1] > terminal_size[1]:
        args.size = -1, terminal_size[1]-1 *args.debug
        args.size = get_video_size(args)
    
    return size

def read_video(args: Args, start_time=None):
    """
    Reads a video file and returns a list of frames using ffmpeg
    """

    size = args.size
    vfilters, afilters, ffmpeg = process_ffmpeg_args(args.ffmpeg)
    if args.tempo != 1:
        vfilters += f',setpts={1/args.tempo}*PTS' if vfilters else f'setpts={1/args.tempo}*PTS'
        atempo = args.tempo
        while atempo > 2:
            atempo /= 2
            afilters += ',atempo=2' if afilters else 'atempo=2'
        while atempo < 0.5:
            atempo *= 2
            afilters += ',atempo=2' if afilters else 'atempo=2'
        afilters += f',atempo={atempo}' if afilters else f'atempo={atempo}'
    
    if args.start_time:
        ffmpeg += f' -ss {args.start_time/args.tempo}'

    vidp = subprocess.Popen(
        f'ffmpeg -i "{args.video_path}" -pix_fmt bgr24 -vf fps={args.fps},{vfilters+"," if vfilters else ""}scale={size[0]}:{size[1]} {ffmpeg} -f rawvideo -',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL)
    vida = subprocess.Popen(
        f'ffmpeg -i "{args.video_path}" -vn -f wav {f"-af {afilters}" if afilters else ""} {ffmpeg} -',
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
        while 'nan' in (data := p.stderr.read(69).decode()):
            pass
        sys.stdout.write('\033[2J')
        yield data
        if start_time is not None:
            start_time[0] = time.time()
        while data:=vidp.stdout.read(3*size[0]*size[1]):
            data = np.frombuffer(data, dtype='uint8').reshape((size[1], size[0], 3))
            # data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            yield data
    finally:
        vidp.stdout.close()
        vidp.kill()
        vida.stdout.close()
        vida.kill()
        if audio:
            p.kill()

def show_video(args: Args):
    args.fps = get_vid_fps(args.video_path) * args.tempo if args.fps is None else args.fps
    path_to_self = os.path.dirname(os.path.realpath(__file__))
    ascii_map = load_ascii_map(os.path.join(path_to_self,'ascii_darkmap.dat'))
    start = [time.time()]
    freq = args.freq
    console_height = os.get_terminal_size()[1]
    get_video_size(args)
    displayed_frame_count = 0
    sys.stdout.write(f'\033[{(console_height-args.size[1]+1)//2}H\033[s')
    skipped_frames = 0
    interlace_start = 0
    for i, frame in enumerate(read_video(args, start_time=start)):
        if frame is None:
            break
        if i+3 < args.fps * (time.time() - start[0]):
            skipped_frames += 1
            continue
        
        # colorless_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        s = 'n'.join('█' * frame.shape[1] for _ in range(frame.shape[0])) if args.no_ascii else cimg2ascii(frame, ascii_map)
        freq = cget_freq(freq, args.min_freq, frame, args.max_chars, interlace_start, args.interlace)
        ns = cinsert_color(s, frame, freq, interlace_start, args.interlace) if not args.colorless else s
        
        while i > args.fps * (time.time() - start[0]):
            pass
        
        displayed_frame_count += 1
        interlace_start = (interlace_start + 1) % args.interlace

        data_str = f'freq: {str(freq).ljust(3)}dropped frames: {str(skipped_frames).ljust(5)}fps: {str(round(displayed_frame_count/(time.time()-start[0]), 2)).ljust(6)}strlen: {len(ns)}'
        sys.stdout.write(f'\033[u\033[0m{data_str}\033[E{ns}'
                         if args.debug else f'\033[u{ns}')


if __name__ == "__main__":
    
    # Add args if python run in debug mode
    if sys.gettrace():
        sys.argv.extend(['D:\RIFE app\Waifu Racks\out.mkv','-d', '-t', '2.5', '--ffmpeg', '-t', '10'])
    usage = f'''\nUsage: vid2ascii.py <input_file> [options]\n
    Options:
        -h : Show this help message

        -d, --debug : Show debug information (default: {Args.debug})
        -i [n], --interlace [n] : Interlace the video by n (optional) rows. 1 means no interlace. If only `-i` is specified, default to 2. Must be greater than 0 (default: {Args.interlace})
        -m <max_chars>, --max-chars <max_chars> : Maximum number of characters to display (default: {Args.max_chars})
        -mf <min_freq>, --min-freq <min_freq> : Minimum threshold for color change to display (default: {Args.min_freq})
        --no-ascii : Don't use ascii characters to represent the video (default: {Args.no_ascii})
        --no-color : Don't use color in the output (default: {Args.colorless})
        -r <fps>, --fps <fps> : Framerate of the output video (default: video's framerate)
        -s <width>:<height> : Size/scale of the output video (default: {SIZE[0]}:{SIZE[1]})
        -ss : Skip to specified time in the video in seconds. (default: {Args.start_time})
        -t <tempo>, --tempo <tempo> : Tempo of the output video (ex. 1x speed, 2x speed, 1.75x speed) (default: {Args.tempo})
        
        --ffmpeg [...] : All commands after this will be passed to ffmpeg'''
    if len(sys.argv) < 2 and not os.path.isfile(Args.video_path):
        print(usage)
        sys.exit(0)
    Args.video_path = sys.argv.pop(1) if len(sys.argv) > 1 else Args.video_path
    if Args.video_path in ('-h', '--help'):
        print(usage)
        sys.exit(0)
    if not os.path.isfile(Args.video_path):
        print(f'File "{Args.video_path}" not found')
        sys.exit(1)
    try:
        while len(sys.argv) > 1:
            val = sys.argv.pop(1)
            if val in ('-d', '--debug'):
                Args.debug = True
            elif val in ('-i', '--interlace'):
                if len(sys.argv) > 1 and sys.argv[1][0] != '-':
                    Args.interlace = int(sys.argv.pop(1))
                    if Args.interlace < 1:
                        sys.stdout.write('Interlace must be greater than 0\n')
                        raise Exception
                else:
                    Args.interlace = 2
            elif val in ('-m', '--max-chars'):
                Args.max_chars = int(sys.argv.pop(1))
                if Args.max_chars < 1:
                    sys.stdout.write('Max chars must be greater than 0\n')
                    raise Exception
            elif val in ('-mf', '--min-freq'):
                Args.min_freq = int(sys.argv.pop(1))
            elif val in ('--no-ascii'):
                Args.no_ascii = True
            elif val in ('--no-color'):
                Args.colorless = True
            elif val in ('-r', '--fps'):
                Args.fps = int(sys.argv.pop(1))
                if Args.fps < 1:
                    sys.stdout.write('FPS must be greater than 0\n')
                    raise Exception
            elif val in ('-s'):
                Args.size = tuple(map(int, sys.argv.pop(1).split(':')))
                if len(Args.size) != 2:
                    sys.stdout.write('Invalid size\n')
            elif val in ('-ss'):
                Args.start_time = float(sys.argv.pop(1))
                if Args.start_time < 0:
                    sys.stdout.write('Start time must be greater than 0\n')
                    raise Exception
            elif val in ('-t', '--tempo'):
                Args.tempo = float(sys.argv.pop(1))
                if Args.tempo <= 0:
                    sys.stdout.write('Tempo must be greater than 0\n')
                    raise Exception
            elif val in ('--ffmpeg'):
                Args.ffmpeg = sys.argv[1:]
                break
            else:
                print(usage)
                sys.exit(0)
    except Exception as e:
        print(usage)
        sys.exit(0)

    try:
        sys.stdout.write('\033[?25l\033[2J')
        show_video(Args)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(f'\033[0m\033[?25h\033[9999H')
