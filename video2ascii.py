from img2ascii import img2ascii, load_ascii_map
import sys
import os
from PIL import Image
from time import perf_counter
from threading import Thread
import shutil
import vlc


SIZE = 266, 61


def read_video(video_path, fps=None):
    """
    Reads a video file and returns a list of frames using ffmpeg
    """
    try:
        if not os.path.exists('temp'):
            os.mkdir('temp')
            os.system(f'ffmpeg -y -i "{video_path}" -vn temp/out.flac')
            os.system(f'ffmpeg -y -i "{video_path}" -q:v 31 {f"-vf fps={fps},scale=800:-1 " if fps else ""}temp/%08d.jpg')

        try:
            img = 1
            p = vlc.MediaPlayer('temp/out.flac')
            while not os.path.exists(f'temp/{img:08d}.jpg'):
                pass
            p.play()
            while True:
                yield Image.open(f'temp/{img:08d}.jpg')
                # os.remove(f'temp/{img:08d}.jpg')
                img += 1
        except FileNotFoundError:
            pass
        except Exception as e:
            print(e)
        finally:
            p.release()
    finally:
        pass
        # shutil.rmtree('temp')


def show_video(path, fps=55, size=SIZE):
    ascii_map = load_ascii_map('ascii_darkmap.dat')
    start = perf_counter()
    for frame in read_video(path, fps):
        frame = frame.convert('L')
        s = img2ascii(frame, ascii_map, size)
        while ((end := perf_counter()) - start) * fps < 1:
            pass
        start = end
        # st = perf_counter()
        Thread(target=sys.stdout.write, args=(s,)).start()
        # print(perf_counter() - st)
        # break


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.argv.append('crf18.mp4')
        sys.argv.append('80')
        sys.argv.append(SIZE[0])
        sys.argv.append(SIZE[1])
    elif len(sys.argv) < 4:
        sys.argv.append(SIZE[0])
        sys.argv.append(SIZE[1])
    video_path = sys.argv[1]
    fps = float(sys.argv[2])
    show_video(video_path, fps, (int(sys.argv[3]), int(sys.argv[4])))
