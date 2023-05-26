@echo off

set COMPRESS_FLAGS="--ultra-brute"
set COMPRESS=1

if exist dist rmdir /s /q dist

py -3.11 -m nuitka --standalone --include-data-files=ascii_darkmap.dat=ascii_darkmap.dat video2ascii.py

mkdir dist

move video2ascii.dist\_ctypes.pyd dist/
move video2ascii.dist\ascii_darkmap.dat dist/
move video2ascii.dist\cimg2ascii.pyd dist/
move video2ascii.dist\libffi-8.dll dist/
move video2ascii.dist\python3.dll dist/
move video2ascii.dist\python311.dll dist/
move video2ascii.dist\video2ascii.exe dist/
move video2ascii.dist\cv2 dist\
move video2ascii.dist\numpy dist\

if %COMPRESS%==0 goto skip_compress
start "cv2" upx %COMPRESS_FLAGS% dist\cv2\cv2.pyd
start "cv2 ffmpeg" upx %COMPRESS_FLAGS% dist\cv2\*.dll
start  "np libopenblas" upx %COMPRESS_FLAGS% dist\numpy\.libs\*
start "vid2asc" upx %COMPRESS_FLAGS% dist\video2ascii.exe
start "dlls" upx %COMPRESS_FLAGS% dist\*.dll
start "pyds" upx %COMPRESS_FLAGS% dist\*.pyd
start "np core" upx %COMPRESS_FLAGS% dist\numpy\core\*
start "np fft" upx %COMPRESS_FLAGS% dist\numpy\fft\*
start "np linalg" upx %COMPRESS_FLAGS% dist\numpy\linalg\*
start "np random" upx %COMPRESS_FLAGS% dist\numpy\random\*

:skip_compress
rmdir /s /q video2ascii.dist
rmdir /s /q video2ascii.build

rem Use powershell to create a zip file
powershell Compress-Archive -Path dist/* -DestinationPath video2ascii.zip -Force -CompressionLevel Optimal
