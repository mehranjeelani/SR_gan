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
parser.add_argument('--data_set',default='Pascal_VOC',type=str,help='training dataset',choices=['Pascal_VOC','DIVFK'])
parser.add_argument('--case',default = 'train_without_noise',type = str,help = 'noise type')


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
   
    netG = Generator(UPSCALE_FACTOR)
    netG.apply(weights_init)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    netD.apply(weights_init)
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    
    generator_criterion = GeneratorLoss()
    discriminator_criterion =  nn.BCELoss()
    
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
    
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))    
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    torch.autograd.set_detect_anomaly(True)
   
    for epoch in range(1, NUM_EPOCHS + 1):

        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
        netG.train()
        netD.train()
        
        for data, target in train_bar:
           
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size
            label = torch.ones((batch_size,)).cuda()
            ############################
            # (1) Update D network: maximize log(D(x))+log(1-D(G(z)))
            ###########################
            real_img = target
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = data
            if torch.cuda.is_available():
                z = z.cuda()
            
             
            netD.zero_grad()
            real_out = netD(real_img).view(-1)

            # Calculate loss on HR batch
            errD_real = discriminator_criterion(real_out, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = real_out.mean().item()
            ## Train with SR
            # Generate batch of SR
            fake_img = netG(z)
            label.fill_(0)
            # Classify SR batch with D
            fake_out = netD(fake_img.detach()).view(-1)
            # Calculate D's loss on the SR batch
            errD_fake = discriminator_criterion(fake_out, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = fake_out.mean().item()
            # Add the gradients from the HR and SR batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()
            ##################################################################
            #  Update G network: minimize [-log(D(G(z))) + Perceptual + MSE] #
            ##################################################################
            netG.zero_grad()
            fake_out= netD(fake_img).view(-1)
            errG = generator_criterion(fake_out, fake_img, real_img,batch_size)
            
            errG.backward()
            optimizerG.step()
           

            # loss for current batch before optimization 
            running_results['g_loss'] += errG.item() * batch_size
            running_results['d_loss'] += errD.item()* batch_size
            running_results['d_score'] += D_x* batch_size
            running_results['g_score'] += D_G_z1 * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))
    
        netG.eval()
        out_path = 'training_results/'+DATASET+'/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)
        
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))
        # save model parameters
        if best_model == None or valing_results['psnr'] +valing_results['ssim']> best_model:
            best_model = valing_results['psnr']+valing_results['ssim']
            best_gen = netG
            path = 'epochs/'+DATASET+'/'
            if not os.path.exists(path):
                os.makedirs(path)
            torch.save(netG.state_dict(), path+'netG_%s_epoch_%d_upscale_%d.pth' % (CASE,epoch,UPSCALE_FACTOR))
            #torch.save(netD.state_dict(),path+'netD_%s_epoch_%d_upscale_%d.pth' % (CASE,epoch,UPSCALE_FACTOR))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
    

out_path = 'statistics/'+DATASET+'/'
if not os.path.exists(out_path):
    os.makedirs(out_path)
data_frame = pd.DataFrame(
data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
index=range(1, NUM_EPOCHS + 1))
data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results_'+CASE+'.csv', index_label='Epoch')
#Save results for best model:



netG = best_gen
if torch.cuda.is_available():
    netG = netG.cuda()
    #netG.load_state_dict(torch.load('epochs/'+DATASET+'/'+'netG_%s_epoch_%d_upscale_%d.pth'%(CASE,best_epoch,UPSCALE_FACTOR)))

out_path = 'training_results/'+DATASET+'/SRF_' + str(UPSCALE_FACTOR) + '/'+CASE+'/'
if not os.path.exists(out_path):
    os.makedirs(out_path)
netG.eval()
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
        sr = netG(lr)

    
        val_images.extend(
                [display_transform()(val_lr.squeeze(0)),display_transform()(val_hr_restore.squeeze(0)), display_transform()(sr.data.cpu().squeeze(0)),
                    display_transform()(hr.data.cpu().squeeze(0))])
        val_bar.set_description('Generating SR images for validation set') 
    val_images = torch.stack(val_images)
        #print('total number of images are {} total number of chunks are {}'.format(val_images.size(0),val_images.size(0)//15))
    val_images = torch.split(val_images, 20)
        #print('bfore tqdm val images shape is {}'.format(val_images[0].shape))
    val_save_bar = tqdm(val_images[:-1], desc='[saving validation SR images]')
        #print('images per chunk:{}'.format( val_save_bar[0].shape))


    index = 1
    for image in val_save_bar:
            #print('image shpae is {}'.format(image.shape))
        image = utils.make_grid(image, nrow=4, padding=5)
        utils.save_image(image, out_path + 'index_%d_%s.png' % (index,CASE), padding=5)
        index += 1
        

