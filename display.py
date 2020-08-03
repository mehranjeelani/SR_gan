import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn

import pytorch_ssim
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
       nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
       nn.init.normal_(m.weight.data, 1.0, 0.02)
       nn.init.constant_(m.bias.data, 0)

#torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=80, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=200, type=int, help='train epoch number')
parser.add_argument('--data_set',default='Pascal_VOC',type=str,help='training dataset')
parser.add_argument('--case',default = 'train_with_noise_std_1',type = str,help = 'noise type')

lr = 0.0002
beta1 = 0.5
best_model = None
best_gen = None
if __name__ == '__main__':
    opt = parser.parse_args()
    
    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    DATASET = opt.data_set
    CASE = opt.case
    train_set = TrainDatasetFromFolder('data/train/'+DATASET+'/train_HR', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
    val_set = ValDatasetFromFolder('data/train/'+DATASET+'/valid_HR', upscale_factor=UPSCALE_FACTOR)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)






out_path = 'training_results/'+DATASET+'/SRF_' + str(UPSCALE_FACTOR) + '/'+CASE+'/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

with torch.no_grad():
    val_bar = tqdm(val_loader)
    val_images = []
    for val_lr, val_hr_restore, val_hr in val_bar:
        batch_size = val_lr.size(0)
        lr = val_lr
        hr = val_hr
        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()
       

    
        val_images.extend(
                [display_transform()(val_lr.squeeze(0)),display_transform()(val_hr_restore.squeeze(0)),
                    display_transform()(hr.data.cpu().squeeze(0))])
        val_bar.set_description('Generating SR images for validation set') 
    val_images = torch.stack(val_images)
        #print('total number of images are {} total number of chunks are {}'.format(val_images.size(0),val_images.size(0)//15))
    val_images = torch.split(val_images, 15)
        #print('bfore tqdm val images shape is {}'.format(val_images[0].shape))
    val_save_bar = tqdm(val_images[:-1], desc='[saving validation SR images]')
        #print('images per chunk:{}'.format( val_save_bar[0].shape))


    index = 1
    for image in val_save_bar:
            #print('image shpae is {}'.format(image.shape))
        image = utils.make_grid(image, nrow=3, padding=5)
        utils.save_image(image, out_path + 'index_%d_%s.png' % (index,CASE), padding=5)
        index += 1
        

