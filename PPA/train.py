# encoding: utf-8

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from model.layer import KLDivLoss

from data.sampler import IdentitySampler
from model.generator import Generator
from torch.utils.tensorboard import SummaryWriter
import os
from math import log10
from utils.image_metric import ssim
from model.baseline import Teacher,Normalize

global ITER
ITER = 0


def create_supervised_trainer(model, generator, optimizer, criterion, cfg, metric, logger, teacher,device=None,):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (dict - class:`torch.optim.Optimizer`): the optimizer to use
        criterion (dict - class:loss function): the loss function to use
        cetner_loss_weight (float, optional): the weight for cetner_loss_weight
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    train_mode = cfg.MODEL.MODE
    output_mode = cfg.MODEL.OUTPUT_MODE
    cetner_loss_weight = cfg.SOLVER.CENTER_LOSS_WEIGHT
    r1 = [cfg.MODEL.VAL_R1]
    r1_ir = [cfg.MODEL.VAL_R1]
    epsilon_psnr = cfg.MODEL.PSNR
    epsilon_ssim = cfg.MODEL.SSIM
    correct = 0

    def _update(engine, batch):
        model.train()

        if 'center' in optimizer.keys():
            optimizer['center'].zero_grad()
            
        img_rgb_0, img_rgb_1, img_rgb_M_0, img_rgb_M_1, img_ir, img_ir_M, target_rgb, target_ir= batch

        img_rgb=torch.cat((img_rgb_0, img_rgb_1,), 0)
        img_rgb_M=torch.cat((img_rgb_M_0, img_rgb_M_1,), 0)

       
        img_rgb=img_rgb.to(device)
        
        anonymous_enough = metric['psnr_OP_rgb'] < metric['psnr_OM_rgb'] + epsilon_psnr and metric['ssim_OP_rgb'] < metric['ssim_OM_rgb'] + epsilon_ssim 
        anonymous_enough = anonymous_enough and metric['psnr_OP_ir'] < metric['psnr_OM_ir'] + epsilon_psnr and metric['ssim_OP_ir'] < metric['ssim_OM_ir'] + epsilon_ssim
        
        if anonymous_enough and metric['r1'] > cfg.MODEL.VAL_R1 and metric['r1_ir'] > cfg.MODEL.VAL_R1:
            if metric['r1'] > r1[0] and metric['r1_ir'] > r1_ir[0]:
                logger.info("update generator")
                generator.load_param(cfg, 'G')
                generator.eval()
                print('迭代gan生成')
                r1[0] = metric['r1']
                r1_ir[0] = metric['r1_ir']
            img_rgb_M, img_ir_M = generator(img_rgb, img_ir)
        img_rgb_M = img_rgb_M.to(device) if torch.cuda.device_count() >= 1 else img_rgb_M
        img_rgb_M = img_rgb_M.detach()
        
        img_ir_M = img_ir_M.to(device) if torch.cuda.device_count() >= 1 else img_ir_M
        img_ir_M = img_ir_M.detach()  
          
        targets = torch.cat((target_rgb, target_rgb, target_ir), 0)   
        targets = targets.to(device) if torch.cuda.device_count() >= 1 else targets
        target_rgb = target_rgb.to(device) if torch.cuda.device_count() >= 1 else target_rgb
        target_ir = target_ir.to(device) if torch.cuda.device_count() >= 1 else target_ir
        

        score_O, feat_O, score_P, feat_P = model(img_rgb, img_rgb_M, img_ir, img_ir_M)
        # score_O, feat_O = model(img_rgb, img_rgb_M, img_ir, img_ir_M)
        
        
        loss_C, batch_acc = criterion['reid'](score_O, feat_O,score_P, feat_P, targets)

        
        _,feat_O_t=teacher(img_rgb.to(device),img_ir.to(device),0)

        ######################逐渐减小ts的权重######################
        alpha = 1.0 - 0.5 * (engine.state.epoch / cfg.SOLVER.MAX_EPOCHS)
        alpha = max(0.5,min(1.0,alpha))
        loss_ts = alpha * criterion['ts'](feat_O_t,Normalize(2)(feat_O))
        
        model.set_requires_grad(model.netD1, True) 
        if cfg.MODEL.SCHEMA == "dual":
            model.set_requires_grad(model.netD2, True) 
        loss_D1 = criterion['D1']()
        loss_D2 = criterion['D2']()
        optimizer['d1'].zero_grad()
        if cfg.MODEL.SCHEMA == "dual":
            optimizer['d2'].zero_grad()
        loss_D = loss_D1 + loss_D2
        loss_D.backward()
        optimizer['d1'].step()
        if cfg.MODEL.SCHEMA == "dual":
            optimizer['d2'].step()
        model.set_requires_grad(model.netD1, False) 
        if cfg.MODEL.SCHEMA == "dual":
            model.set_requires_grad(model.netD2, False) 
        
        optimizer['c'].zero_grad()
        
        loss_G_gan = criterion['G_gan']() * 1
        loss_G_l1 = criterion['G_l1']() * 100
        
        loss_G = loss_G_gan + loss_G_l1
        loss_pose = torch.tensor(0.0, device=device)
        # if cfg.MODEL.POSE_GUIDE == 'on' and engine.state.epoch >=cfg.MODEL.POSE_STARTEPOCH:
        if cfg.MODEL.POSE_GUIDE == 'on':
            loss_pose = cfg.MODEL.POSE_LAMBDA * criterion['G_guidance']()
            loss_G = loss_G + loss_pose
        #loss_reid = 0 * loss_ts + loss_C
        loss_reid = cfg.MODEL.TS_SCALE * loss_ts + loss_C
        loss_GC = cfg.MODEL.TO_SCALE * loss_G + loss_reid
        
        # loss_air = torch.tensor(0.0, device=device)
        # if cfg.MODEL.AIR == 'on':
        #     loss_air = criterion['air'](model.real_rgb_O, model.fake_rgb_P)  # 先做 RGB 分支；如需 IR，可再加一项    
        #     # loss_air = loss_air + criterion['air'](model.real_ir_O, model.fake_ir_P)
        #     loss_GC = loss_GC + cfg.MODEL.AIR_LAMBDA * loss_air

        loss_dcma = torch.tensor(0.0, device=device)
        if cfg.MODEL.DCMA == 'on':
            loss_dcma = criterion['dcma']()
            loss_GC = loss_GC + cfg.MODEL.DCMA_LAMBDA * loss_dcma
        optimizer['g'].zero_grad()
        
        loss_GC.backward()    
        optimizer['g'].step()
        optimizer['c'].step()
        loss = loss_D + loss_GC
        
        if 'center' in optimizer.keys():
            for param in criterion['center'].parameters():
                param.grad.data *= (1. / cetner_loss_weight)
            optimizer['center'].step()

        
        # compute acc
        if output_mode == 'both':
            acc = correct
        elif output_mode == 'protected':
            acc = correct
        else:
            acc = correct
        if output_mode == 'both':
            return loss.item(), acc, loss_G_gan.item(), loss_G_l1.item(), loss_D1.item(), loss_D2.item(), loss_C.item(), loss_ts.item(),loss_pose.item(),loss_dcma.item()
        elif output_mode == 'protected':
            return loss.item(), acc, loss_G_gan.item(), loss_G_l1.item(), loss_D1.item(), loss_D2.item(), loss_C.item(), loss_ts.item(), loss_pose.item(),loss_dcma.item()
        else:
            return loss.item(), acc, loss_G_gan.item(), loss_G_l1.item(), loss_D1.item(), loss_D2.item(), loss_C.item(), loss_ts.item(), loss_pose.item(),loss_dcma.item()


    return Engine(_update)


def do_train(
        cfg,
        model,
        data_loader,
        optimizer,
        schedulers,
        criterion,
        num_query,
        num_gallery,
        start_epoch,
        details,
        num_classes
):
    # model.eval()
    model.train()
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    val_period = cfg.SOLVER.VAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("Cross-SecureReID")
    logger.info("Start training")

    
    generator=Generator(cfg)
    # generator.load_param(cfg)
    teacher= Teacher(num_classes, cfg.MODEL.GENERALIZED_MEAN_POOL)
    teacher.load_param(cfg,teacher)
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    

    metric = {'r1': 0.0,'r1_ir': 0.0, 'r1_max':0.0,\
              'psnr_OM_rgb':0.0, 'ssim_OM_rgb':0.0, 'psnr_OP_rgb':100.0, 'ssim_OP_rgb':1.0,\
              'psnr_OM_ir':0.0, 'ssim_OM_ir':0.0, 'psnr_OP_ir':100.0, 'ssim_OP_ir':1.0,\
              }
    test_metric = {'r1': 0.0,'r1_ir':0.0}
    trainer = create_supervised_trainer(model, generator, optimizer, criterion, cfg, metric, logger,teacher, device=device)

    # tensorboard writer
    loss_G_L1_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "loss", "G_L1"))
    loss_G_GAN_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "loss", "G_GAN"))
    loss_D1_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "loss", "D1"))
    loss_D2_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "loss", "D2"))
    loss_C_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "loss", "C"))
    loss_ts_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "loss", "ts"))
    
    r1_irO_rgbO_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "metrics", "r1_irO_rgbO"))
    map_irO_rgbO_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "metrics", "mAP_irO_rgbO"))
    minp_irO_rgbO_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "metrics", "mINP_irO_rgbO"))
    
    r1_irO_rgbP_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "metrics", "r1_irO_rgbP"))
    map_irO_rgbP_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "metrics", "mAP_irO_rgbP"))
    minp_irO_rgbP_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "metrics", "mINP_irO_rgbP"))
    
    r1_irP_rgbO_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "metrics", "r1_irP_rgbO"))
    map_irP_rgbO_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "metrics", "mAP_irP_rgbO"))
    minp_irP_rgbO_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "metrics", "mINP_irP_rgbO"))
    
    r1_irP_rgbP_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "metrics", "r1_irP_rgbP"))
    map_irP_rgbP_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "metrics", "mAP_irP_rgbP"))
    minp_irP_rgbP_writer = SummaryWriter(os.path.join(cfg.OUTPUT_DIR, 'runs', "metrics", "mINP_irP_rgbP"))

     
    valing_results = {'batch_sizes_rgb': 0, 'batch_sizes_ir': 0, \
    'mse_OP_rgb': 0, 'ssims_OP_rgb': 0, 'psnr_OP_rgb': 0, 'ssim_OP_rgb': 0, \
    'mse_OM_rgb': 0, 'ssims_OM_rgb': 0, 'psnr_OM_rgb': 0, 'ssim_OM_rgb': 0, \
    'mse_OP_ir': 0, 'ssims_OP_ir': 0, 'psnr_OP_ir': 0, 'ssim_OP_ir': 0, \
    'mse_OM_ir': 0, 'ssims_OM_ir': 0, 'psnr_OM_ir': 0, 'ssim_OM_ir': 0, \
        }
    
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, score_function=lambda x: checkpoint_period, n_saved=10, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer_g': optimizer['g'],
                                                                     'optimizer_c': optimizer['c'],
                                                                     'optimizer_d1': optimizer['d1'],
                                                                     'optimizer_d2': optimizer['d2']
                                                                    #  'optimizer_d3': optimizer['d3'],
                                                                    #  'center_param': criterion['center'],
                                                                    #  'optimizer_center': optimizer['center']
                                                                    })
    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    if cfg.MODEL.OUTPUT_MODE == 'both':
        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
        RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')
        RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'loss_G_gan')
        RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'loss_G_l1')
        RunningAverage(output_transform=lambda x: x[4]).attach(trainer, 'loss_D1')
        RunningAverage(output_transform=lambda x: x[5]).attach(trainer, 'loss_D2')
        RunningAverage(output_transform=lambda x: x[6]).attach(trainer, 'loss_C')
        RunningAverage(output_transform=lambda x: x[7]).attach(trainer, 'loss_ts')
        RunningAverage(output_transform=lambda x: x[8]).attach(trainer, 'loss_pose')  # 新增
        RunningAverage(output_transform=lambda x: x[9]).attach(trainer, 'loss_dcma')  # 新增


    elif cfg.MODEL.OUTPUT_MODE == 'protected':
        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
        RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')
        RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'loss_G_gan')
        RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'loss_G_l1')
        RunningAverage(output_transform=lambda x: x[4]).attach(trainer, 'loss_D1')
        RunningAverage(output_transform=lambda x: x[5]).attach(trainer, 'loss_D2')
        RunningAverage(output_transform=lambda x: x[6]).attach(trainer, 'loss_C')
        RunningAverage(output_transform=lambda x: x[7]).attach(trainer, 'loss_ts')
        RunningAverage(output_transform=lambda x: x[8]).attach(trainer, 'loss_pose')  # 新增
        RunningAverage(output_transform=lambda x: x[9]).attach(trainer, 'loss_dcma')  # 新增

    else:
        RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
        RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')
        RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'loss_G_gan')
        RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'loss_G_l1')
        RunningAverage(output_transform=lambda x: x[4]).attach(trainer, 'loss_D1')
        RunningAverage(output_transform=lambda x: x[5]).attach(trainer, 'loss_D2')
        RunningAverage(output_transform=lambda x: x[6]).attach(trainer, 'loss_C')
        RunningAverage(output_transform=lambda x: x[7]).attach(trainer, 'loss_ts')
        RunningAverage(output_transform=lambda x: x[8]).attach(trainer, 'loss_pose')  # 新增
        RunningAverage(output_transform=lambda x: x[9]).attach(trainer, 'loss_dcma')  # 新增


    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch
        
    @trainer.on(Events.EPOCH_STARTED)
    def prepare_dataLoader(engine):
        print('==> Preparing Data Loader...')
        sampler=IdentitySampler(data_loader['trainset'].train_color_label, data_loader['trainset'].train_thermal_label, cfg.DATALOADER.NUM_POS, cfg.DATALOADER.NUM_INSTANCE)
        data_loader['trainset'].cIndex=sampler.index1
        data_loader['trainset'].tIndex=sampler.index2
        loader_batch=cfg.DATALOADER.NUM_POS * cfg.DATALOADER.NUM_INSTANCE
        data_loader['train'] = DataLoader(
            data_loader['trainset'],
            batch_size=loader_batch,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=True
        )
        print(engine.state.epoch)
        print(data_loader['trainset'].cIndex)
        print(data_loader['trainset'].tIndex)

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        print('==> adjust_learning_rate...')
        for scheduler in schedulers:
            scheduler.step()
        
        def adjust_lr(optimizer, epoch):
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            b_lr = 0.1
            if epoch < 10:
                lr = b_lr * (epoch + 1) / 10
            elif epoch >= 10 and epoch < 20:
                lr = b_lr
            elif epoch >= 20 and epoch < 50:
                lr = b_lr * 0.1
            elif epoch >= 50:
                lr = b_lr * 0.01

            optimizer.param_groups[0]['lr'] = 0.1 * lr
            for i in range(len(optimizer.param_groups) - 1):
                optimizer.param_groups[i + 1]['lr'] = lr

            return lr
        
        current_lr = adjust_lr(optimizer['c'], engine.state.epoch)
        
        print('current_lr_C',current_lr)
        
        


    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(data_loader['train']),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                #schedulers[0].get_lr()[0]))
                                0))
            if cfg.MODEL.OUTPUT_MODE == 'both':
                logger.info("Loss_G_gan: {:.3f}, Loss_G_l1: {:.3f}, Loss_D1: {:.3f}, Loss_D2: {:.3f}, Loss_C: {:.3f},Loss_ts: {:.3f},Loss_pose: {:.3f},Loss_dcma: {:.3f}"
                        .format(engine.state.metrics['loss_G_gan'],engine.state.metrics['loss_G_l1'],
                                engine.state.metrics['loss_D1'],  engine.state.metrics['loss_D2'],  engine.state.metrics['loss_C'],
                                engine.state.metrics['loss_ts'], engine.state.metrics['loss_pose'],engine.state.metrics['loss_dcma']))
                loss_G_L1_writer.add_scalar('loss', engine.state.metrics['loss_G_l1'], engine.state.epoch)
                loss_G_GAN_writer.add_scalar('loss', engine.state.metrics['loss_G_gan'], engine.state.epoch)
                loss_D1_writer.add_scalar('loss',engine.state.metrics['loss_D1'], engine.state.epoch)
                loss_D2_writer.add_scalar('loss',engine.state.metrics['loss_D2'], engine.state.epoch)
                loss_C_writer.add_scalar('loss',engine.state.metrics['loss_C'], engine.state.epoch)
                loss_ts_writer.add_scalar('loss',engine.state.metrics['loss_ts'], engine.state.epoch)
            elif cfg.MODEL.OUTPUT_MODE == 'protected':
                logger.info("Loss_G_gan: {:.3f}, Loss_G_l1: {:.3f}, Loss_D1: {:.3f}, Loss_D2: {:.3f}, Loss_C: {:.3f}, Loss_ts: {:.3f},Loss_pose: {:.3f}"
                    .format(engine.state.metrics['loss_G_gan'],engine.state.metrics['loss_G_l1'],
                            engine.state.metrics['loss_D1'],  engine.state.metrics['loss_D2'],  engine.state.metrics['loss_C'],
                            engine.state.metrics['loss_ts'], engine.state.metrics['loss_pose']))
                loss_G_L1_writer.add_scalar('loss', engine.state.metrics['loss_G_l1'], engine.state.epoch)
                loss_G_GAN_writer.add_scalar('loss', engine.state.metrics['loss_G_gan'], engine.state.epoch)
                loss_D1_writer.add_scalar('loss',engine.state.metrics['loss_D1'], engine.state.epoch)
                loss_D2_writer.add_scalar('loss',engine.state.metrics['loss_D2'], engine.state.epoch)
                loss_C_writer.add_scalar('loss',engine.state.metrics['loss_C'], engine.state.epoch)
                loss_ts_writer.add_scalar('loss',engine.state.metrics['loss_ts'], engine.state.epoch)
            else:
                logger.info("Loss_G_gan: {:.3f}, Loss_G_l1: {:.3f}, Loss_D1: {:.3f}, Loss_D2: {:.3f}, Loss_C: {:.3f}, Loss_ts: {:.3f},Loss_pose: {:.3f}"
                    .format(engine.state.metrics['loss_G_gan'],engine.state.metrics['loss_G_l1'],
                            engine.state.metrics['loss_D1'],  engine.state.metrics['loss_D2'],  engine.state.metrics['loss_C'],
                            engine.state.metrics['loss_ts'], engine.state.metrics['loss_pose']))
                loss_G_L1_writer.add_scalar('loss', engine.state.metrics['loss_G_l1'], engine.state.epoch)
                loss_G_GAN_writer.add_scalar('loss', engine.state.metrics['loss_G_gan'], engine.state.epoch)
                loss_D1_writer.add_scalar('loss',engine.state.metrics['loss_D1'], engine.state.epoch)
                loss_D2_writer.add_scalar('loss',engine.state.metrics['loss_D2'], engine.state.epoch)
                loss_C_writer.add_scalar('loss',engine.state.metrics['loss_C'], engine.state.epoch)
                loss_ts_writer.add_scalar('loss',engine.state.metrics['loss_ts'], engine.state.epoch)
        if len(data_loader['train']) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            data_loader['train'].batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        r1 = 0.0
        if engine.state.epoch % eval_period == 0:
            cmc1, mAP1, mINP1, cmc2, mAP2, mINP2, cmc3, mAP3, mINP3, cmc4, mAP4, mINP4 = test(model, num_gallery, num_query, data_loader, cfg, details,valing_results)
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info('test')
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

            r1_irO_rgbO_writer.add_scalar('R1', cmc1[0], engine.state.epoch)
            map_irO_rgbO_writer.add_scalar('mAP', mAP1, engine.state.epoch)
            minp_irO_rgbO_writer.add_scalar('mINP', mINP1, engine.state.epoch)
            
            r1_irO_rgbP_writer.add_scalar('R1', cmc2[0], engine.state.epoch)
            map_irO_rgbP_writer.add_scalar('mAP', mAP2, engine.state.epoch)
            minp_irO_rgbP_writer.add_scalar('mINP', mINP2, engine.state.epoch)

            r1_irP_rgbO_writer.add_scalar('R1', cmc3[0], engine.state.epoch)
            map_irP_rgbO_writer.add_scalar('mAP', mAP3, engine.state.epoch)
            minp_irP_rgbO_writer.add_scalar('mINP', mINP3, engine.state.epoch)
            
            r1_irP_rgbP_writer.add_scalar('R1', cmc4[0], engine.state.epoch)
            map_irP_rgbP_writer.add_scalar('mAP', mAP4, engine.state.epoch)
            minp_irP_rgbP_writer.add_scalar('mINP', mINP4, engine.state.epoch)  
                                
            r1 = cmc2[0]
            r1_ir = cmc3[0]
        
            if r1 > test_metric['r1'] and r1_ir > test_metric['r1_ir']:
                test_metric['r1'] = r1
                test_metric['r1_ir'] = r1_ir
                logger.info('saving the test best model (epoch %d)' % (engine.state.epoch))
                model.save_networks('epoch_' + 'best')


        if engine.state.epoch % eval_period == 0: 
            # save the latest model
            logger.info('saving the latest model (epoch %d)' % (engine.state.epoch))
            model.save_networks('epoch_' + 'latest')
            logger.info('rgb_PSNR')
            logger.info('OP:{0} OM:{1}'.format(valing_results['psnr_OP_rgb'], valing_results['psnr_OM_rgb']))
            logger.info('rgb_SSIM')
            logger.info('OP:{0} OM:{1}'.format(valing_results['ssim_OP_rgb'], valing_results['ssim_OM_rgb']))
            
            logger.info('ir_PSNR')
            logger.info('OP:{0} OM:{1}'.format(valing_results['psnr_OP_ir'], valing_results['psnr_OM_ir']))
            logger.info('ir_SSIM')
            logger.info('OP:{0} OM:{1}'.format(valing_results['ssim_OP_ir'], valing_results['ssim_OM_ir'])) 
                       
            logger.info("Saving images......")
            metric['psnr_OP_rgb'] = valing_results['psnr_OP_rgb']
            metric['ssim_OP_rgb'] = valing_results['ssim_OP_rgb']
            metric['psnr_OM_rgb'] = valing_results['psnr_OM_rgb']
            metric['ssim_OM_rgb'] = valing_results['ssim_OM_rgb']
            
            
            metric['psnr_OP_ir'] = valing_results['psnr_OP_ir']
            metric['ssim_OP_ir'] = valing_results['ssim_OP_ir']
            metric['psnr_OM_ir'] = valing_results['psnr_OM_ir']
            metric['ssim_OM_ir'] = valing_results['ssim_OM_ir']
                      
            metric['r1'] = r1
            metric['r1_ir'] = r1_ir
            
            # image_numpy = util.tensor2im(model.real_O[0], mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            # util.save_image(image_numpy, '{0}/{1}_real_O.jpg'.format(cfg.OUTPUT_DIR, engine.state.epoch))
            # image_numpy = util.tensor2im(model.real_M[0], mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            # util.save_image(image_numpy, '{0}/{1}_real_M.jpg'.format(cfg.OUTPUT_DIR, engine.state.epoch))
            # image_numpy = util.tensor2im(model.fake_P_encrypted[0], mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            # util.save_image(image_numpy, '{0}/{1}_fake_P_encrypted.jpg'.format(cfg.OUTPUT_DIR, engine.state.epoch))
            # image_numpy = util.tensor2im(model.fake_P[0], mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            # util.save_image(image_numpy, '{0}/{1}_fake_P.jpg'.format(cfg.OUTPUT_DIR, engine.state.epoch))
            # image_numpy = util.tensor2im(model.fake_R[0], mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
            # util.save_image(image_numpy, '{0}/{1}_fake_R.jpg'.format(cfg.OUTPUT_DIR, engine.state.epoch))



    trainer.run(data_loader['train'], max_epochs=epochs)
    
    

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
            input = input.to(cfg.MODEL.DEVICE)
            input_M = input_M.to(cfg.MODEL.DEVICE)
            feat_OI, feat_PI = model(input, input, input, input, 0)
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
            input = input.to(cfg.MODEL.DEVICE)
            input_M = input_M.to(cfg.MODEL.DEVICE)
            feat_OI, feat_PI = model(input, input, input, input, 1)
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