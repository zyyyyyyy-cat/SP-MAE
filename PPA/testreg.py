
import argparse
import os
import sys
import time
from PIL import Image
import numpy as np
import torch
from torch.backends import cudnn
sys.path.append('.')
from config import cfg
from data.build import make_data_loader
from model import build_model
from utils.logger import setup_logger
from math import log10
from utils.image_metric import ssim



def fliplr(img):
    '''flip horizontal'''
    # inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    # img_flip = img.index_select(3,inv_idx)
    img_flip = img
    return img_flip


def test(model,ngall, nquery, data_loader,cfg,details,valing_results):
    # switch to evaluation mode
    model.eval()
    
    mean, std = cfg.INPUT.PIXEL_MEAN , cfg.INPUT.PIXEL_STD
    mean = torch.as_tensor(mean)
    std = torch.as_tensor(std)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1).to(cfg.MODEL.DEVICE)
    if std.ndim == 1:
        std = std.view(-1, 1, 1).to(cfg.MODEL.DEVICE)
    
    print('Extracting Gallery Feature...')
    ptr = 0
    gall_feat_OI = np.zeros((ngall, cfg.MODEL.CENTER_FEAT_DIM))
    gall_feat_PI = np.zeros((ngall, cfg.MODEL.CENTER_FEAT_DIM))
    with torch.no_grad():
        for batch_idx, (input, input_M, label) in enumerate(data_loader['gall_loader']):
            batch_num = input.size(0)
            input = fliplr(input).to(cfg.MODEL.DEVICE)
            input_M = fliplr(input_M).to(cfg.MODEL.DEVICE)
            feat_OI, feat_PI = model(input, input, input, input, 1)   #ir2vis is 0   , vis2ir is 1
            gall_feat_OI[ptr:ptr + batch_num, :] = feat_OI.detach().cpu().numpy()
            gall_feat_PI[ptr:ptr + batch_num, :] = feat_PI.detach().cpu().numpy()
            ptr = ptr + batch_num
            
            valing_results['batch_sizes_rgb'] += batch_num
            input_OI = input.mul(std).add(mean)
            input_MI = input_M.mul(std).add(mean)
            img_PI = model.fake_P.mul(std).add(mean)
            
            batch_mse_OM = ((input_OI - input_MI) ** 2).data.mean()
            batch_mse_OP = ((input_OI - img_PI) ** 2).data.mean()
            
            valing_results['mse_OM_rgb'] += batch_mse_OM * batch_num
            valing_results['mse_OP_rgb'] += batch_mse_OP * batch_num
            
            batch_ssim_OM = ssim(input_OI, input_MI, device=cfg.MODEL.DEVICE).data
            batch_ssim_OP = ssim(input_OI, img_PI, device=cfg.MODEL.DEVICE).data
            
            valing_results['ssims_OM_rgb'] += batch_ssim_OM * batch_num
            valing_results['ssims_OP_rgb'] += batch_ssim_OP * batch_num
            
            valing_results['psnr_OM_rgb'] = 10 * log10(1 / (valing_results['mse_OM_rgb'] / valing_results['batch_sizes_rgb']))
            valing_results['psnr_OP_rgb'] = 10 * log10(1 / (valing_results['mse_OP_rgb'] / valing_results['batch_sizes_rgb']))
            
            valing_results['ssim_OM_rgb'] = valing_results['ssims_OM_rgb'] / valing_results['batch_sizes_rgb']
            valing_results['ssim_OP_rgb'] = valing_results['ssims_OP_rgb'] / valing_results['batch_sizes_rgb']

    # switch to evaluation
    model.eval()
    print('Extracting Query Feature...')
    ptr = 0
    query_feat_OI = np.zeros((nquery, cfg.MODEL.CENTER_FEAT_DIM))
    query_feat_PI = np.zeros((nquery, cfg.MODEL.CENTER_FEAT_DIM))
    with torch.no_grad():
        for batch_idx, (input, input_M, label) in enumerate(data_loader['query_loader']):
            batch_num = input.size(0)
            input = fliplr(input).to(cfg.MODEL.DEVICE)
            input_M = fliplr(input_M).to(cfg.MODEL.DEVICE)
            feat_OI, feat_PI = model(input, input, input, input, 0)   #ir2vis is 1  , vis2ir is 0
            query_feat_OI[ptr:ptr + batch_num, :] = feat_OI.detach().cpu().numpy()
            query_feat_PI[ptr:ptr + batch_num, :] = feat_PI.detach().cpu().numpy()
            ptr = ptr + batch_num
            
            valing_results['batch_sizes_ir'] += batch_num
            input_OI = input.mul(std).add(mean)
            input_MI = input_M.mul(std).add(mean)
            img_PI = model.fake_P.mul(std).add(mean)
            
            batch_mse_OM = ((input_OI - input_MI) ** 2).data.mean()
            batch_mse_OP = ((input_OI - img_PI) ** 2).data.mean()
            
            valing_results['mse_OM_ir'] += batch_mse_OM * batch_num
            valing_results['mse_OP_ir'] += batch_mse_OP * batch_num
            
            batch_ssim_OM = ssim(input_OI, input_MI, device=cfg.MODEL.DEVICE).data
            batch_ssim_OP = ssim(input_OI, img_PI, device=cfg.MODEL.DEVICE).data
            
            valing_results['ssims_OM_ir'] += batch_ssim_OM * batch_num
            valing_results['ssims_OP_ir'] += batch_ssim_OP * batch_num
            
            valing_results['psnr_OM_ir'] = 10 * log10(1 / (valing_results['mse_OM_ir'] / valing_results['batch_sizes_ir']))
            valing_results['psnr_OP_ir'] = 10 * log10(1 / (valing_results['mse_OP_ir'] / valing_results['batch_sizes_ir']))
            
            valing_results['ssim_OM_ir'] = valing_results['ssims_OM_ir'] / valing_results['batch_sizes_ir']
            valing_results['ssim_OP_ir'] = valing_results['ssims_OP_ir'] / valing_results['batch_sizes_ir']
            
            


    # compute the similarity
    distmat_OO = np.matmul(query_feat_OI, np.transpose(gall_feat_OI))
    distmat_OP = np.matmul(query_feat_OI, np.transpose(gall_feat_PI))
    distmat_PO = np.matmul(query_feat_PI, np.transpose(gall_feat_OI))
    distmat_PP = np.matmul(query_feat_PI, np.transpose(gall_feat_PI))
    
    
    # evaluation
    if cfg.DATASETS.NAMES == 'sysu':
        cmc1, mAP1, mINP1 = eval_sysu(-distmat_OO, details['query_label'], details['gall_label'], details['query_cam'], details['gall_cam'])
        cmc2, mAP2, mINP2 = eval_sysu(-distmat_OP, details['query_label'], details['gall_label'], details['query_cam'], details['gall_cam'])
        cmc3, mAP3, mINP3 = eval_sysu(-distmat_PO, details['query_label'], details['gall_label'], details['query_cam'], details['gall_cam'])
        cmc4, mAP4, mINP4 = eval_sysu(-distmat_PP, details['query_label'], details['gall_label'], details['query_cam'], details['gall_cam'])        
    elif cfg.DATASETS.NAMES == 'regdb':
        cmc1, mAP1, mINP1 = eval_regdb(-distmat_OO, details['query_label'], details['gall_label'])
        cmc2, mAP2, mINP2 = eval_regdb(-distmat_OP, details['query_label'], details['gall_label'])
        cmc3, mAP3, mINP3 = eval_regdb(-distmat_PO, details['query_label'], details['gall_label'])
        cmc4, mAP4, mINP4 = eval_regdb(-distmat_PP, details['query_label'], details['gall_label'])
        
    return cmc1, mAP1, mINP1, cmc2, mAP2, mINP2, cmc3, mAP3, mINP3, cmc4, mAP4, mINP4

def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank = 50):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)
        
        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]
        
        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])
        
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q   # standard CMC
    
    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return new_all_cmc, mAP, mINP

def eval_regdb(distmat, q_pids, g_pids, max_rank = 50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    
    # only two cameras
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2* np.ones(num_g).astype(np.int32)
    
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return all_cmc, mAP, mINP


parser = argparse.ArgumentParser(description="Cross-SecureReID Re-ID Baseline")
parser.add_argument(
    "--config_file", default="", help="path to config file", type=str
)
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                    nargs=argparse.REMAINDER)

args = parser.parse_args()

num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

if args.config_file != "":
    cfg.merge_from_file(args.config_file)
cfg.merge_from_list(args.opts)
cfg.freeze()

output_dir = cfg.OUTPUT_DIR
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

logger = setup_logger("Cross-SecureReID", output_dir, 0)
logger.info("Using {} GPUS".format(num_gpus))
logger.info(args)

if args.config_file != "":
    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, 'r') as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
logger.info("Running with config:\n{}".format(cfg))

if cfg.MODEL.DEVICE == "cuda":
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID  # new add by gu
cudnn.benchmark = True

# cfg.defrost()
# cfg.MODEL.PRETRAIN_DIR = 'asdadsa'
# cfg.freeze()
# print(cfg.MODEL.PRETRAIN_DIR)
pre_dir=cfg.MODEL.PRETRAIN_DIR

all_results = {
    'psnr_OP_rgb': 0, 'ssim_OP_rgb': 0, \
    'psnr_OM_rgb': 0, 'ssim_OM_rgb': 0, \
    'psnr_OP_ir': 0, 'ssim_OP_ir': 0, \
    'psnr_OM_ir': 0, 'ssim_OM_ir': 0, \
    }

for trial in range(10):
    test_trial=trial+1
    cfg.defrost()
    cfg.MODEL.PRETRAIN_DIR = pre_dir + 'trial{}'.format(test_trial)
    cfg.MODEL.TRIAL = test_trial
    cfg.freeze()
    print('=========',cfg.MODEL.PRETRAIN_DIR)
    data_loader, nquery, ngall, num_classes, details=make_data_loader(cfg)


    query_loader = data_loader['query_loader']

    model = build_model(cfg, len(np.unique(details['gall_label'])))

    model.to(cfg.MODEL.DEVICE)

    logger.info("Evaluate Only")
    model.load_param(cfg)

    valing_results = {'batch_sizes_rgb': 0, 'batch_sizes_ir': 0, \
        'mse_OP_rgb': 0, 'ssims_OP_rgb': 0, 'psnr_OP_rgb': 0, 'ssim_OP_rgb': 0, \
        'mse_OM_rgb': 0, 'ssims_OM_rgb': 0, 'psnr_OM_rgb': 0, 'ssim_OM_rgb': 0, \
        'mse_OP_ir': 0, 'ssims_OP_ir': 0, 'psnr_OP_ir': 0, 'ssim_OP_ir': 0, \
        'mse_OM_ir': 0, 'ssims_OM_ir': 0, 'psnr_OM_ir': 0, 'ssim_OM_ir': 0, \
            }

    cmc1, mAP1, mINP1, cmc2, mAP2, mINP2, cmc3, mAP3, mINP3, cmc4, mAP4, mINP4 = test(model,ngall, nquery, data_loader,cfg,details,valing_results)

    if trial == 0:
    #         logger.info('rgb_PSNR')
    # logger.info('OP:{0} OM:{1}'.format(valing_results['psnr_OP_rgb'], valing_results['psnr_OM_rgb']))
    # logger.info('rgb_SSIM')
    # logger.info('OP:{0} OM:{1}'.format(valing_results['ssim_OP_rgb'], valing_results['ssim_OM_rgb']))

    # logger.info('ir_PSNR')
    # logger.info('OP:{0} OM:{1}'.format(valing_results['psnr_OP_ir'], valing_results['psnr_OM_ir']))
    # logger.info('ir_SSIM')
    # logger.info('OP:{0} OM:{1}'.format(valing_results['ssim_OP_ir'], valing_results['ssim_OM_ir'])) 
        all_results['psnr_OP_rgb']=valing_results['psnr_OP_rgb']
        all_results['psnr_OM_rgb']=valing_results['psnr_OM_rgb']
        all_results['ssim_OP_rgb']=valing_results['ssim_OP_rgb']
        all_results['ssim_OM_rgb']=valing_results['ssim_OM_rgb']
        all_results['psnr_OP_ir']=valing_results['psnr_OP_ir']
        all_results['psnr_OM_ir']=valing_results['psnr_OM_ir']
        all_results['ssim_OP_ir']=valing_results['ssim_OP_ir']
        all_results['ssim_OM_ir']=valing_results['ssim_OM_ir']
        all_cmc1 = cmc1
        all_mAP1 = mAP1
        all_mINP1 = mINP1
        all_cmc2 = cmc2
        all_mAP2 = mAP2
        all_mINP2 = mINP2
        all_cmc3 = cmc3
        all_mAP3 = mAP3
        all_mINP3= mINP3
        all_cmc4 = cmc4
        all_mAP4 = mAP4
        all_mINP4 = mINP4
    else:
        all_results['psnr_OP_rgb'] = all_results['psnr_OP_rgb'] + valing_results['psnr_OP_rgb']
        all_results['psnr_OM_rgb'] = all_results['psnr_OM_rgb'] + valing_results['psnr_OM_rgb']
        all_results['ssim_OP_rgb'] = all_results['ssim_OP_rgb'] + valing_results['ssim_OP_rgb']
        all_results['ssim_OM_rgb'] = all_results['ssim_OM_rgb'] + valing_results['ssim_OM_rgb']
        all_results['psnr_OP_ir'] = all_results['psnr_OP_ir'] + valing_results['psnr_OP_ir']
        all_results['psnr_OM_ir'] = all_results['psnr_OM_ir'] + valing_results['psnr_OM_ir']
        all_results['ssim_OP_ir'] = all_results['ssim_OP_ir'] + valing_results['ssim_OP_ir']
        all_results['ssim_OM_ir'] = all_results['ssim_OM_ir'] + valing_results['ssim_OM_ir']
        all_cmc1 = all_cmc1 + cmc1
        all_mAP1 = all_mAP1 + mAP1
        all_mINP1 = all_mINP1 + mINP1
        all_cmc2 = all_cmc2 + cmc2
        all_mAP2 = all_mAP2 + mAP2
        all_mINP2 = all_mINP2 + mINP2
        all_cmc3 = all_cmc3 + cmc3
        all_mAP3 = all_mAP3 + mAP3
        all_mINP3 = all_mINP3 + mINP3
        all_cmc4 = all_cmc4 + cmc4
        all_mAP4 = all_mAP4 + mAP4
        all_mINP4 = all_mINP4 + mINP4
    
    logger.info('test trial-{}'.format(test_trial))
    logger.info('Q:origin images G:origin images:')
    logger.info("mINP: {:.1%}".format(mINP1))
    logger.info("mAP: {:.1%}".format(mAP1))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc1[r - 1]))
        
    logger.info('Q:origin images G:protected images:')
    logger.info("mINP: {:.1%}".format(mINP2))
    logger.info("mAP: {:.1%}".format(mAP2))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc2[r - 1]))
        
    logger.info('Q:protected images G:origin images:')
    logger.info("mINP: {:.1%}".format(mINP3))
    logger.info("mAP: {:.1%}".format(mAP3))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc3[r - 1]))
        
    logger.info('Q:protected images G:protected images:')
    logger.info("mINP: {:.1%}".format(mINP4))
    logger.info("mAP: {:.1%}".format(mAP4))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc4[r - 1]))   

    logger.info('rgb_PSNR')
    logger.info('OP:{0} OM:{1}'.format(valing_results['psnr_OP_rgb'], valing_results['psnr_OM_rgb']))
    logger.info('rgb_SSIM')
    logger.info('OP:{0} OM:{1}'.format(valing_results['ssim_OP_rgb'], valing_results['ssim_OM_rgb']))

    logger.info('ir_PSNR')
    logger.info('OP:{0} OM:{1}'.format(valing_results['psnr_OP_ir'], valing_results['psnr_OM_ir']))
    logger.info('ir_SSIM')
    logger.info('OP:{0} OM:{1}'.format(valing_results['ssim_OP_ir'], valing_results['ssim_OM_ir'])) 
    
all_results['psnr_OP_rgb'] = all_results['psnr_OP_rgb'] /test_trial
all_results['psnr_OM_rgb'] = all_results['psnr_OM_rgb'] /test_trial
all_results['ssim_OP_rgb'] = all_results['ssim_OP_rgb'] /test_trial
all_results['ssim_OM_rgb'] = all_results['ssim_OM_rgb'] /test_trial
all_results['psnr_OP_ir'] = all_results['psnr_OP_ir'] /test_trial
all_results['psnr_OM_ir'] = all_results['psnr_OM_ir'] /test_trial
all_results['ssim_OP_ir'] = all_results['ssim_OP_ir'] /test_trial
all_results['ssim_OM_ir'] = all_results['ssim_OM_ir'] /test_trial
all_cmc1 = all_cmc1 /test_trial
all_mAP1 = all_mAP1 /test_trial
all_mINP1 = all_mINP1 /test_trial
all_cmc2 = all_cmc2 /test_trial
all_mAP2 = all_mAP2 /test_trial
all_mINP2 = all_mINP2 /test_trial
all_cmc3 = all_cmc3 /test_trial
all_mAP3 = all_mAP3 /test_trial
all_mINP3 = all_mINP3 /test_trial
all_cmc4 = all_cmc4 /test_trial
all_mAP4 = all_mAP4 /test_trial
all_mINP4 = all_mINP4 /test_trial

logger.info('All Average:')
logger.info('Q:origin images G:origin images:')
logger.info("mINP: {:.1%}".format(all_mINP1))
logger.info("mAP: {:.1%}".format(all_mAP1))
for r in [1, 5, 10]:
    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, all_cmc1[r - 1]))
    
logger.info('Q:origin images G:protected images:')
logger.info("mINP: {:.1%}".format(all_mINP2))
logger.info("mAP: {:.1%}".format(all_mAP2))
for r in [1, 5, 10]:
    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, all_cmc2[r - 1]))
    
logger.info('Q:protected images G:origin images:')
logger.info("mINP: {:.1%}".format(all_mINP3))
logger.info("mAP: {:.1%}".format(all_mAP3))
for r in [1, 5, 10]:
    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, all_cmc3[r - 1]))
    
logger.info('Q:protected images G:protected images:')
logger.info("mINP: {:.1%}".format(all_mINP4))
logger.info("mAP: {:.1%}".format(all_mAP4))
for r in [1, 5, 10]:
    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, all_cmc4[r - 1]))   

logger.info('rgb_PSNR')
logger.info('OP:{0} OM:{1}'.format(all_results['psnr_OP_rgb'], all_results['psnr_OM_rgb']))
logger.info('rgb_SSIM')
logger.info('OP:{0} OM:{1}'.format(all_results['ssim_OP_rgb'], all_results['ssim_OM_rgb']))

logger.info('ir_PSNR')
logger.info('OP:{0} OM:{1}'.format(all_results['psnr_OP_ir'], all_results['psnr_OM_ir']))
logger.info('ir_SSIM')
logger.info('OP:{0} OM:{1}'.format(all_results['ssim_OP_ir'], all_results['ssim_OM_ir'])) 
    