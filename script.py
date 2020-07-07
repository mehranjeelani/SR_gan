import os
import shutil

from PIL import Image
path = './data/train_HR'



with open('train.txt', "r", encoding="utf-8") as file:
    
    
    names = [line.strip() for line in file]
    index = int(len(names ) *0.7)
    remaining = len(names)-index
    half_remaining = remaining//2
    for count in range(half_remaining):
        shutil.move(path+'/'+names[count]+'.jpg', './data/test/Pascal_VOC/')
    for count in range(half_remaining,remaining):

        shutil.move(path+'/'+names[count]+'.jpg', './data/valid_HR/')
        
        
'''
image = Image.open('2007_000175.jpg')
h,w=image.size
image2 = image.resize([h//4,w//4])
image2.save('test2.jpg')
image2 = image.resize([h//4,w//4],Image.ANTIALIAS)
image2.save('test2_wo_aliasing.jpg')
'''




