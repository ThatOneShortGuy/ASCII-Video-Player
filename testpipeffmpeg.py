import subprocess
import numpy as np
import cv2

SIZE = 160, 42
file = 'crf18.mp4'

p = subprocess.Popen(f'ffmpeg -i "{file}" -vf scale={SIZE[0]}:{SIZE[1]} -pix_fmt rgb24 -f rawvideo -', stdout=subprocess.PIPE, bufsize=3*SIZE[0]*SIZE[1])
while True:
    data = p.stdout.read(3*SIZE[0]*SIZE[1])
    if not data:
        break
    data = np.frombuffer(data, dtype='uint8').reshape((SIZE[1], SIZE[0], 3))
    # cv2.imshow('frame', data)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
# cv2.destroyAllWindows()
p.stdout.close()
