import argparse
import os
from math import log10

import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TestDatasetFromFolder, display_transform
from model import Generator
parser = argparse.ArgumentParser(description='Test Benchmark Datasets')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='netG_train_noise_std_0.125_epoch_59_upscale_4.pth', type=str, help='generator model epoch name')
parser.add_argument('--data_set',default='DIVFK',type=str,help='testing dataset')
parser.add_argument('--case',default = 'train_with_noise',type = str,help = 'noise type')
parser.add_argument('--model_used',default = 'Pascal_VOC',type = str,help = 'testing model')


opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name
DATASET = opt.data_set
CASE = opt.case
MODEL_USED = opt.model_used
results = {DATASET: {'psnr': [], 'ssim': [],'psnr_bicubic':[],'ssim_bicubic':[]}}#, 'Set14': {'psnr': [], 'ssim': []}, 'BSD100': {'psnr': [], 'ssim': []},
           #'Urban100': {'psnr': [], 'ssim': []}, 'SunHays80': {'psnr': [], 'ssim': []}}

model = Generator(UPSCALE_FACTOR)
if torch.cuda.is_available():
    model = model.cuda()
model.load_state_dict(torch.load('epochs/'+MODEL_USED+'/' + MODEL_NAME))

test_set = TestDatasetFromFolder('data/test/'+DATASET+'/', upscale_factor=UPSCALE_FACTOR)
test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)
test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')

out_path = 'benchmark_results/'+DATASET+'/SRF_' + str(UPSCALE_FACTOR) + '/'+CASE+'/'

if not os.path.exists(out_path):
    os.makedirs(out_path)
model.eval()
test_images = []
with torch.no_grad():
    for image_name, lr_image, hr_restore_image, hr_image in test_bar:
        
        image_name = image_name[0]
        

        if torch.cuda.is_available():
            lr_image = lr_image.cuda()
            hr_image = hr_image.cuda()
            hr_restore_image = hr_restore_image.cuda()

        sr_image = model(lr_image)
        mse = ((hr_image - sr_image) ** 2).data.mean()
        psnr = 10 * log10(1 / mse)
        ssim = pytorch_ssim.ssim(sr_image, hr_image).item()
        #print('psnr is {}\tssim is {}'.format(psnr,ssim))
        #ssim = pytorch_ssim.ssim(sr_image, hr_image).item()
        mse_bicubic = ((hr_image - hr_restore_image) ** 2).data.mean()
        psnr_bicubic = 10 * log10(1 / mse_bicubic)
        ssim_bicubic = pytorch_ssim.ssim(hr_restore_image, hr_image).item()
        #print('mse bicubic is {}\tssim bicubic is {}'.format(mse_bicubic,ssim_bicubic))
        

        # save psnr\ssim

        results[DATASET]['psnr'].append(psnr)
        results[DATASET]['ssim'].append(ssim)
        results[DATASET]['psnr_bicubic'].append(psnr_bicubic)
        results[DATASET]['ssim_bicubic'].append(ssim_bicubic)
        test_images.extend(
                         [display_transform()(lr_image.data.cpu().squeeze(0)),display_transform()(hr_restore_image.data.cpu().squeeze(0)), display_transform()(sr_image.data.cpu().squeeze(0)),
                 display_transform()(hr_image.data.cpu().squeeze(0))])
        test_bar.set_description('Generating SR images for test set') 
    test_images = torch.stack(test_images)
        #print('total number of images are {} total number of chunks are {}'.format(val_images.size(0),val_images.size(0)//15))
    test_images = torch.split(test_images, 20)
        #print('bfore tqdm val images shape is {}'.format(val_images[0].shape))
    test_save_bar = tqdm(test_images[:-1], desc='[saving test SR images]')
        #print('images per chunk:{}'.format( val_save_bar[0].shape))


    index = 1
    for image in test_save_bar:
            #print('image shpae is {}'.format(image.shape))
        image = utils.make_grid(image, nrow=4, padding=5)
        utils.save_image(image, out_path + 'index_%d_%s.png' % (index,CASE), padding=5)
        index += 1
   

out_path = 'statistics/'+DATASET+'/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

saved_results = {'psnr': [], 'ssim': [],'psnr_bicubic':[],'ssim_bicubic':[]}
for item in results.values():
    psnr = np.array(item['psnr'])
    ssim = np.array(item['ssim'])
    psnr_bicubic = np.array(item['psnr_bicubic'])
    ssim_bicubic = np.array(item['ssim_bicubic'])
    '''
    if (psnr.size == 0) or (ssim.size == 0) or (psnr_bicubic.size==0) or (ssim_bicubic.size==0):
        psnr = 'No data'
        ssim = 'No data'
        psnr_bicubic = 'No data'
        ssim_bicubic = 'No data'
    '''
    
    psnr = psnr.mean()
    ssim = ssim.mean()
    psnr_bicubic = psnr_bicubic.mean()
    ssim_bicubic = ssim_bicubic.mean()
    saved_results['psnr'].append(psnr)
    saved_results['ssim'].append(ssim)
    saved_results['psnr_bicubic'].append(psnr_bicubic)
    saved_results['ssim_bicubic'].append(ssim_bicubic)

data_frame = pd.DataFrame(saved_results, results.keys())
data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_test_results_'+CASE+'.csv', index_label='DataSet')

