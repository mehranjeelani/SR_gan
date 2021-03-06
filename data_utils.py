from os import listdir
from os.path import join
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.mean = mean
        self.std = std
                                                                                                                                            
    def __call__(self, tensor):
        #print('adding gaussian noise')
        tensor =  tensor + torch.randn(tensor.size()) * self.std + self.mean
        tensor = torch.clamp(tensor,min=0.0,max=1.0)
        return tensor
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])

def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.NEAREST),
        ToTensor()
        
    ])
def train_lr_transform_noisy():
    return Compose([AddGaussianNoise(0.,0.125)])

def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])
def noisy_transform_val():
    return Compose([
        ToTensor(),
        AddGaussianNoise(0.,0.125),
        ToPILImage()
    ])

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)
        self.lr_transform_noisy  = train_lr_transform_noisy()
        self.upscale_factor = upscale_factor
    def __getitem__(self, index):
        hr_image =self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        lr_image_noisy = self.lr_transform_noisy(lr_image)

      
        
        
      
        #return lr_image, hr_image
        return lr_image_noisy, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.NEAREST)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        lr_image_noisy = noisy_transform_val()(lr_image)
        hr_restore_img = hr_scale(lr_image)
        hr_restore_img_noisy = hr_scale(lr_image_noisy)
        #return (ToTensor()(lr_image)), ToTensor()(hr_restore_img), ToTensor()(hr_image)
        return (ToTensor()(lr_image_noisy)), ToTensor()(hr_restore_img_noisy), ToTensor()(hr_image)

	

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        #self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        #self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.NEAREST)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        lr_image_noisy = noisy_transform_val()(lr_image) 
        hr_restore_img = hr_scale(lr_image)
        hr_restore_img_noisy = hr_scale(lr_image_noisy)
 
        image_name = self.image_filenames[index].split('/')[-1]
        return image_name, (ToTensor()(lr_image_noisy)), ToTensor()(hr_restore_img_noisy), ToTensor()(hr_image)
        #return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)
    
    def __len__(self):
        return len(self.image_filenames)


