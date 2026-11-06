                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       # encoding: utf-8

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .data_loader import SYSUData, TestData, RegDBData
from .sampler import IdentitySampler
from .data_manager import process_query_sysu, process_gallery_sysu, process_test_regdb


def make_data_loader(cfg):
    start = time.time()
    if cfg.DATASETS.NAMES == 'sysu':
        dataset = SYSUData(cfg.DATASETS.ROOT_DIR, type=cfg.INPUT.TYPE, radius=cfg.INPUT.RADIUS)
    elif cfg.DATASETS.NAMES == 'regdb':
        dataset = RegDBData(cfg.DATASETS.ROOT_DIR, trial= cfg.MODEL.TRIAL,type=cfg.INPUT.TYPE, radius=cfg.INPUT.RADIUS)
        
    sampler=IdentitySampler(dataset.train_color_label, dataset.train_thermal_label, cfg.DATALOADER.NUM_POS, cfg.DATALOADER.NUM_INSTANCE)
    dataset.cIndex=sampler.index1
    dataset.tIndex=sampler.index2
    
    
    num_classes = len(np.unique(dataset.train_color_label))
    num_workers = cfg.DATALOADER.NUM_WORKERS
    loader_batch=cfg.DATALOADER.NUM_POS * cfg.DATALOADER.NUM_INSTANCE
    

    data_loader={}
    data_loader['trainset'] = dataset
    data_loader['train'] = DataLoader(
        dataset,
        batch_size=loader_batch,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True
    )
    
    print('data_loader-train Time:\t {:.3f}'.format(time.time() - start))
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    normalize])
    if cfg.DATASETS.NAMES == 'sysu':
        query_img, query_label, query_cam = process_query_sysu(cfg.DATASETS.ROOT_DIR, mode='all')
        gall_img, gall_label, gall_cam = process_gallery_sysu(cfg.DATASETS.ROOT_DIR, mode='all', trial=cfg.MODEL.TRIAL)
    elif cfg.DATASETS.NAMES == 'regdb':
        query_img, query_label = process_test_regdb(cfg.DATASETS.ROOT_DIR, trial=cfg.MODEL.TRIAL, modal='visible') #原来是thermal
        gall_img, gall_label = process_test_regdb(cfg.DATASETS.ROOT_DIR, trial=cfg.MODEL.TRIAL, modal='thermal')
        query_cam = []
        gall_cam = []
        
    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(128, 256), type=cfg.INPUT.TYPE, radius=cfg.INPUT.RADIUS)
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(128, 256), type=cfg.INPUT.TYPE, radius=cfg.INPUT.RADIUS)
    
    nquery = len(query_label)
    ngall = len(gall_label)
    
    print('Dataset {} statistics:'.format(cfg.DATASETS.NAMES))
    print('  ------------------------------')
    print('  subset   | # ids | # images')
    print('  ------------------------------')
    print('  visible  | {:5d} | {:8d}'.format(num_classes, len(dataset.train_color_label)))
    print('  thermal  | {:5d} | {:8d}'.format(num_classes, len(dataset.train_thermal_label)))
    print('  ------------------------------')
    print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
    print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
    print('  ------------------------------')
    
    data_loader['query_loader'] = DataLoader(queryset, batch_size=16, shuffle=False, num_workers=num_workers)
    data_loader['gall_loader'] = DataLoader(gallset, batch_size=16, shuffle=False, num_workers=num_workers)

    details={}
    details['query_label']=query_label
    details['query_cam']=query_cam
    details['gall_label']=gall_label
    details['gall_cam']=gall_cam
    
    
    return data_loader, nquery, ngall, num_classes, details
