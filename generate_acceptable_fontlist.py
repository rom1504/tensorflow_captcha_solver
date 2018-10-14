from itertools import chain

import glob
from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode


def font_contains_letter_digit(filename):
    print("checking " + filename)
    ttf = TTFont(filename, 0, allowVID=0,
                 ignoreDecompileErrors=True,
                 fontNumber=-1)

    chars = chain.from_iterable([y + (Unicode[y[0]],) for y in x.cmap.items()] for x in ttf["cmap"].tables)

    charset = set()

    for x in chars:
        charset.add(x[0])

    for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
        char = ord(c)
        if char not in charset:
            return False
    return True


fonts = []
with open('fonts.txt', 'a') as the_file:
    for filename in glob.iglob('/usr/share/fonts/truetype/**/*.ttf', recursive=True):
        if font_contains_letter_digit(filename):
            print("adding "+filename)
            the_file.write(filename + "\n")