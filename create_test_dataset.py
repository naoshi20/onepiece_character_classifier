import numpy as np
from PIL import Image
from io import BytesIO
import h5py
import os
import random

img_size = (64, 64)

filenames = os.listdir("test_dataset/imgs") #dataというフォルダにある画像を読み込み
num_files = len(filenames) #画像数を把握し
imgs = []
labels = []
for i in range(num_files):
    imgname = filenames[i]
    try:
        originalImg = Image.open("test_dataset/imgs/" + imgname)
        resizeImg = originalImg.resize(img_size)
        w, h = resizeImg.size
        pixels = []

        img = []
        for y in range(h):
            pixels = []
            for x in range(w):
                pixel = []
                color = resizeImg.getpixel((x, y))
                pixel.append(color[0])
                pixel.append(color[1])
                pixel.append(color[2])
                pixels.append(pixel)
            img.append(pixels)
        imgs.append(img)

        character = imgname.split("_")[-2]
        print(character)
        if character == "ruffy":
            labels.append(0)
        if character == "zoro":
            labels.append(1)
        if character == "others":
            labels.append(2)

    except:
        print("read error:", imgname)
        num_files -= 1
        continue

imgs_array = np.array(imgs)
label_array = np.array(labels)

im_labels = list(zip(imgs, labels))

random.sample(im_labels, len(im_labels)) # 重複なし。　random.shuffle()はNoneを返す。
im_labels = sorted(im_labels, key=lambda k: random.random())

x_test = []
y_test = []

for i in range(30):
    x_test.append(im_labels[i][0])
    y_test.append(im_labels[i][1])

print(len(x_test),len(y_test))

classes = [0, 1, 2]

with h5py.File('test_dataset/dataset/onepiece_' + str(img_size[0]) +'.h5', 'w') as f:

    f.create_dataset('test_set_x', data=x_test)
    f.create_dataset('test_set_y', data=y_test)
    f.create_dataset('list_classes', data=classes)

"""
#画像とラベルの一致を確認。
im1 = x_train[3]
im2 = Image.new("RGB", (h,w))
for x in range(h):
    for y in range(w):
        r = im1[x][y][0]
        g = im1[x][y][1]
        b = im1[x][y][2]
        im2.putpixel((x,y),(r,g,b))
im2.show()
label = y_train[3]
print("This image represents " + str(label) + ".")
"""
