import os
import shutil

from PIL import Image
path = ['./data/train_HR','./data/valid_HR']

'''
for filename in ['train.txt','val.txt']:
    with open(filename, "r", encoding="utf-8") as file:
        if filename == 'train.txt':
            print(len([line for line in file]))
'''
image = Image.open('2007_000175.jpg')
h,w=image.size
image2 = image.resize([h//4,w//4])
image2.save('test2.jpg')
image2 = image.resize([h//4,w//4],Image.ANTIALIAS)
image2.save('test2_wo_aliasing.jpg')





