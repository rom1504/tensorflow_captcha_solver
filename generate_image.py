from PIL import Image, ImageDraw, ImageFont
import random

fonts = [
    ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', 15),
    ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', 15),
    ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf', 13)
]

words = set()
for i in range(0, 10000):
    word = ""
    for j in range(0, 5):
        word += random.choice('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    if word in words:
        continue

    img = Image.new("1", (50, 20))
    img.paste((1), [0, 0, img.size[0], img.size[1]])

    d = ImageDraw.Draw(img)
    d.text((1, 5), word, font=random.choice(fonts), fill=(0))
    img.save('generated_captcha_images2/' + word + '.png')
    words.add(word)