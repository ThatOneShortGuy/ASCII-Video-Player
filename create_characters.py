from PIL import Image, ImageFont, ImageDraw

START = 33
END = 10000

for i in range(START, END):
    blank = Image.new('RGBA', (14, 18), (0, 0, 0, 0))
    img = ImageDraw.Draw(blank)
    font = ImageFont.truetype('Consolas.ttf', 16)
    img.text((2, 0), chr(i), (255, 255, 255), font)
    blank.save(f'ascii\\{i}.png')
