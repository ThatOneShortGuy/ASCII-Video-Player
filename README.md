# ASCII Video Player

![Sample Image](Images/image.png)

# Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [How to compile and run](#how-to-compile-and-run)
  - [Video to ASCII Art](#video-to-ascii-art)
    - [Options](#options)
  - [Image to ASCII Art](#image-to-ascii-art)
    - [Options](#options-1)


# Introduction
This program converts images or videos to colored ASCII art. It uses the in house methods from this directory to create the ASCII art. The characters used for the ASCII art are calculated by using the [create_characters.py](create_characters.py) script with the [create_ascii_darkmap.py](create_ascii_darkmap.py) script. The characters are then stored in the [ascii_darkmap.dat](ascii_darkmap.dat) file. The [img2ascii.py](img2ascii.py) and [video2ascii.py](video2ascii.py) scripts use the [ascii_darkmap.dat](ascii_darkmap.dat) file to convert images or videos to ASCII art. If you don't have the [ascii_darkmap.dat](ascii_darkmap.dat) in your directory, the programs will not work.

# Dependencies and Requirements
- Python 3.8 or higher (tested on 3.9.6)
- cv2
- numpy
- ffmpeg [for the video to ASCII converter] (you will need to have the ffmpeg.exe in your PATH environment)
- ffplay [for the video to ASCII converter] (you will need to have the ffplay.exe in your PATH environment for audio playback)
- psutil [for the video to ASCII converter] (you will need to install this using pip)

To install the dependencies, run the following command:
```bash
pip install -U opencv-python psutil
```

# How to compile and run
You will need a compiler for C. On Windows, this is the [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

Run the following command to compile the program:
```
python "C Funcs\setup.py" build_ext --inplace
```

## Video to ASCII Art
Then you can run the ASCII video player by running the following command:
```shell
python video2ascii.py <video path> [options]
```

### Options
- `-h`: Show help message and exit
- `--colorless`: Don't use color in the output (default: False)
- `-d`, `--debug`: Show debug information (default: False)
- `--no-ascii`: Don't use ascii characters to represent the video (default: False)
- `-r <fps>, --fps <fps>`: Frames per second. The framerate to play the video back at (default: video's framerate)
- `-s <width>:<height>`: Size of the output video. Should be input as "width:height" with no spaces and numbers only. A negative one (-1) in any of the sizes will calculate the best size to maintain the image ratio (default: 160:-1)
- `--ffmpeg [...]`: All commands after this will be passed to ffmpeg video decoder. See [ffmpeg documentation](https://ffmpeg.org/ffmpeg.html) for more information (default: None)

## Image to ASCII Art
Then you can run the image to ASCII art converter by running the following command:
```bash
python img2ascii.py <image path> [options]
```

### Options
- `-h`: Show help message and exit
- `-f <freq>`, `-c <freq>`: Color frequency. The higher the integer, the higher threshold for changing color. (default: 2)
- `-s <width>:<height>`: Size of the output image. Should be input as "width:height" with no spaces and numbers only. A negative one (-1) in any of the sizes will calculate the best size to maintain the image ratio (default: 266:61)