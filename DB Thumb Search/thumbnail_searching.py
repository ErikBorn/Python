import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
from PIL import Image
import numpy as np
# %matplotlib inline
def make_url(num):
    return f"https://fp-zoom-us.s3.amazonaws.com/5303/5303_{num}.JPG"

first = 20000
last = 20500
# last = 98806

def get_img(num):
    num = num.zfill(6)
    url = make_url(num)
    response = requests.get(url)
    pic = response.content
    img = Image.open(io.BytesIO(pic))
    return img

def count_teal(im):
    teal = 0
    r, g, b = (122, 239, 210)
    for pixel in im.getdata():
        if all([
            r - 20 < pixel[0] < r + 20,
            g - 20 < pixel[1] < g + 20,
            b - 20 < pixel[2] < b + 20,
        ]):
            teal += 1
    return teal

def p(im):
    plt.imshow(np.asarray(im))

image_to_teal_count = {}
high_teal = []
for i in range(first, last):
    image = get_img(f"{i}")
    teal_count = count_teal(image)
    image_to_teal_count[i] = teal_count
    if teal_count > 10:
        print(i, teal_count)
        high_teal.append(image)

for im in high_teal:
    plt.figure()
    p(im)