from configparser import Interpolation
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.utils.data as data
from .ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing, ChannelExchange
import torchvision.transforms as transforms
import random
import math
from config import cfg
import torchvision.transforms.functional as TF


class SYSUData(data.Dataset):
    def __init__(self, data_dir, colorIndex = None, thermalIndex = None,type='mosaic', radius=8):
        
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_thermal_image = train_thermal_image
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        self.type = type
        self.radius = radius
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_thermal = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((256, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5)])
            
        self.transform_color = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((256, 128)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(p = 0.1),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5)])
            
        self.transform_color1 = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((256, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelExchange(gray = 2)])
       
    def __getitem__(self, index):

        img1,  target_rgb = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target_ir = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
           
        img_rgb_M = img1.copy()
        img_ir_M = img2.copy()
        img_rgb_M=Image.fromarray(img_rgb_M)
        img_ir_M=Image.fromarray(img_ir_M)
        
        img_rgb_M = TF.resize(img_rgb_M, [256, 128], TF.InterpolationMode.BILINEAR)
        img_ir_M = TF.resize(img_ir_M, [256, 128], TF.InterpolationMode.BILINEAR)
        
        if self.type == 'mosaic':
            do_mosaic(img_rgb_M, self.radius)
            do_mosaic(img_ir_M, self.radius)
        elif self.type == 'blur':
            img_rgb_M = img_rgb_M.filter(ImageFilter.GaussianBlur(radius=int(self.radius)))
            img_ir_M = img_ir_M.filter(ImageFilter.GaussianBlur(radius=int(self.radius)))
        elif self.type == 'noise':
            add_noise = Gaussian_noise(0, self.radius)
            img_rgb_M = add_noise(img_rgb_M)    
            add_noise = Gaussian_noise(0, self.radius)
            img_ir_M = add_noise(img_ir_M)    
            
                     
        img_rgb_0 = self.transform_color(img1)
        img_rgb_1 = self.transform_color1(img1)
        img_ir = self.transform_thermal(img2) 
        img_rgb_M=np.array(img_rgb_M)   
        img_ir_M=np.array(img_ir_M)  
        img_rgb_M_0=self.transform_color(img_rgb_M)  
        img_rgb_M_1=self.transform_color1(img_rgb_M)
        img_ir_M=self.transform_thermal(img_ir_M)

        
        return img_rgb_0, img_rgb_1, img_rgb_M_0, img_rgb_M_1, img_ir, img_ir_M, target_rgb, target_ir

    def __len__(self):
        return len(self.train_color_label)
        
 
class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (128,256),type='mosaic', radius=8):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform
        self.type = type
        self.radius = radius

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        
        img_M = img1.copy()
        img_M=Image.fromarray(img_M)
        img_M = TF.resize(img_M, [256, 128], TF.InterpolationMode.BILINEAR)
        if self.type == 'mosaic':
            do_mosaic(img_M, self.radius)
        elif self.type == 'blur':
            img_M = img_M.filter(ImageFilter.GaussianBlur(radius=int(self.radius)))
        elif self.type == 'noise':
            add_noise = Gaussian_noise(0, self.radius)
            img_M = add_noise(img_M)
        
        img1 = self.transform(img1)
          
        img_M=np.array(img_M) 
        img_M = self.transform(img_M)
        return img1, img_M, target1

    def __len__(self):
        return len(self.test_image)
        
   
    
def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label

def  do_mosaic(frame, neighbor=8):
    fw, fh = frame.size
    draw = ImageDraw.Draw(frame)
    neighbor = int(neighbor)
    for i in range(0, fh, int(neighbor)):  
        for j in range(0, fw, int(neighbor)):
            r, g, b = frame.getpixel((j, i))
            left_up = (j, i)
            width = min(neighbor, fw - j)
            height = min(neighbor, fh - i)
            right_down = (j + width - 1, i + height - 1)
            draw.rectangle((left_up, right_down), fill=(r, g, b), outline=None)


class Gaussian_noise(object):
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        img_ = np.array(img).copy()
        img_ = img_ / 255.0
        noise = np.random.normal(self.mean, self.sigma, img_.shape)
        gaussian_out = img_ + noise
        gaussian_out = np.clip(gaussian_out, 0, 1)
        gaussian_out = np.uint8(gaussian_out * 255)
        # noise = np.uint8(noise*255)
        return Image.fromarray(gaussian_out).convert('RGB')
    
    
class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, colorIndex = None, thermalIndex = None, type='mosaic', radius=8):
        # Load training images (path) and labels
        
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((128, 256), Image.LANCZOS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((128, 256), Image.LANCZOS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        
        self.type = type
        self.radius = radius
        # BGR to RGB
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform_thermal = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((256, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5)])
            
        self.transform_color = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((256, 128)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomGrayscale(p = 0.1),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5)])
            
        self.transform_color1 = transforms.Compose( [
            transforms.ToPILImage(),
            transforms.Pad(10),
            transforms.RandomCrop((256, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ChannelRandomErasing(probability = 0.5),
            ChannelExchange(gray = 2)])

    # def __getitem__(self, index):

    #     img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
    #     img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
    #     img1_0 = self.transform_color(img1)
    #     img1_1 = self.transform_color1(img1)
    #     img2 = self.transform_thermal(img2)

    #     return img1_0, img1_1, img2, target1, target2
    
    def __getitem__(self, index):

        img1,  target_rgb = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target_ir = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
           
        img_rgb_M = img1.copy()
        img_ir_M = img2.copy()
        img_rgb_M=Image.fromarray(img_rgb_M)
        img_ir_M=Image.fromarray(img_ir_M)
        
        img_rgb_M = TF.resize(img_rgb_M, [256, 128], TF.InterpolationMode.BILINEAR)
        img_ir_M = TF.resize(img_ir_M, [256, 128], TF.InterpolationMode.BILINEAR)
        
        if self.type == 'mosaic':
            do_mosaic(img_rgb_M, self.radius)
            do_mosaic(img_ir_M, self.radius)
        elif self.type == 'blur':
            img_rgb_M = img_rgb_M.filter(ImageFilter.GaussianBlur(radius=int(self.radius)))
            img_ir_M = img_ir_M.filter(ImageFilter.GaussianBlur(radius=int(self.radius)))
        elif self.type == 'noise':
            add_noise = Gaussian_noise(0, self.radius)
            img_rgb_M = add_noise(img_rgb_M)    
            add_noise = Gaussian_noise(0, self.radius)
            img_ir_M = add_noise(img_ir_M)    
            
                     
        img_rgb_0 = self.transform_color(img1)
        img_rgb_1 = self.transform_color1(img1)
        img_ir = self.transform_thermal(img2) 
        img_rgb_M=np.array(img_rgb_M)   
        img_ir_M=np.array(img_ir_M)  
        img_rgb_M_0=self.transform_color(img_rgb_M)  
        img_rgb_M_1=self.transform_color1(img_rgb_M)
        img_ir_M=self.transform_thermal(img_ir_M)

        
        return img_rgb_0, img_rgb_1, img_rgb_M_0, img_rgb_M_1, img_ir, img_ir_M, target_rgb, target_ir

    def __len__(self):
        return len(self.train_color_label)