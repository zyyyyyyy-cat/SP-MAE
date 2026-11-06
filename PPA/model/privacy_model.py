import collections
import os
import torch.nn.functional as F
import torch
import torch.nn as nn
from config import cfg
import numpy as np
from . import networks, Baseline
from .baseline import AGW
from .layer import CrossEntropyLabelSmooth, TripletLoss, WeightedRegularizedTriplet, CenterLoss, TripletLoss_ADP, TripletLoss_WRT, TwoStageAnonymizationLoss
from .vit_baseline import make_vit
from .fusion_model import InterleaveFusion2D

class PrivacyModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super(PrivacyModel, self).__init__()

        self.cfg = cfg
        self.train_mode = cfg.MODEL.MODE
        self.output_mode = cfg.MODEL.OUTPUT_MODE
        self.device_id = cfg.MODEL.DEVICE_ID
        str_ids = self.device_id.split(',')
        self.gpu_ids = []
        for id in range(len(str_ids)):
            self.gpu_ids.append(int(str_ids[id]))

        self.isTrain = cfg.TEST.EVALUATE_ONLY == 'off'
        os.environ['CUDA_VISIBLE_DEVICE'] = self.device_id
        # self.device = torch.device(cfg.MODEL.DEVICE) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(cfg.OUTPUT_DIR)  # save all the checkpoints to save_dir

        
        norm = 'batch'
        if self.isTrain:
            if cfg.MODEL.SCHEMA == "dual":
                self.model_names = ['G', 'C', 'D1', 'D2']
            else:
                self.model_names = ['G', 'C', 'D1']
            # self.model_names = ['C']
        else:
            self.model_names = ['G', 'C']
            # self.model_names = ['C']
        # define networks
        input_nc = output_nc = 3
        
        #cfg.MODEL.NGF=64 | cfg.MODEL.NETG='unet_128'
        #cfg.MODEL.NO_DROPOUT='store_true' | cfg.MODEL.INIT_TYPE='normal'
        #cfg.MODEL.INIT_GAIN=0.02
        self.netG = networks.define_G(input_nc, output_nc, cfg.MODEL.NGF, cfg.MODEL.NETG, norm,
                                        not cfg.MODEL.NO_DROPOUT, cfg.MODEL.INIT_TYPE, cfg.MODEL.INIT_GAIN,
                                        self.gpu_ids, cfg.SOLVER.IMS_PER_BATCH, "anonymize", cfg.MODEL.DEVICE)
        # self.netG_ir = networks.define_G(input_nc, output_nc, cfg.MODEL.NGF, cfg.MODEL.NETG, norm,
        #                         not cfg.MODEL.NO_DROPOUT, cfg.MODEL.INIT_TYPE, cfg.MODEL.INIT_GAIN,
        #                         self.gpu_ids, cfg.SOLVER.IMS_PER_BATCH, "anonymize_ir", cfg.MODEL.DEVICE)
        
        if cfg.MODEL.NAME == 'transformer':
            self.netC = make_vit(cfg, num_classes)
        elif cfg.MODEL.NAME == 'baseline':
        
            self.netC = AGW(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NAME,
                                 cfg.MODEL.GENERALIZED_MEAN_POOL, cfg.MODEL.PRETRAIN_CHOICE)
        else:
           self.netC = Baseline(num_classes, cfg.MODEL.GENERALIZED_MEAN_POOL)
        #    if cfg.TEST.EVALUATE_ONLY == 'on':
        #         self.netC.load_param(cfg,self.netC)
        
        self.fusion=InterleaveFusion2D(c_in=cfg.MODEL.CENTER_FEAT_DIM)
        self.fusion.to(cfg.MODEL.DEVICE)
       
        

        ###################### Pose-Guided ######################
        self.mask_head_rgb = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        ).to(cfg.MODEL.DEVICE)
        self.mask_head_ir = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        ).to(cfg.MODEL.DEVICE)
        self.pose_model = None
        if cfg.MODEL.POSE_GUIDE == 'on':
            # import torchvision
            # self.pose_model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
            # self.pose_model.to(cfg.MODEL.DEVICE)
            # self.pose_model.eval()
            # for p in self.pose_model.parameters():
            #     p.requires_grad = False
            from ultralytics import YOLO
            self.pose_model = YOLO('yolov8n-pose.pt')
            self.pose_model.to('cpu')
            # self.pose_model.to(cfg.MODEL.DEVICE)
            # self.pose_model.to(cfg.MODEL.DEVICE)
            # self.pose_model.eval()
            # for p in self.pose_model.parameters():
            #     p.requires_grad = False
        ##################################################################
        
        # self.netC.to(cfg.MODEL.DEVICE)
        if len(self.gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.netC.to(self.gpu_ids[0])
            self.netC = torch.nn.DataParallel(self.netC, self.gpu_ids)  # multi-GPUs
        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            input_nc_D = input_nc + output_nc
            self.netD1 = networks.define_D(input_nc_D, cfg.MODEL.NDF, cfg.MODEL.NETD,
                                            cfg.MODEL.N_LAYERS_D, norm, cfg.MODEL.INIT_TYPE, cfg.MODEL.INIT_GAIN,
                                            self.gpu_ids)
            
            self.netD2 = networks.define_D(input_nc_D, cfg.MODEL.NDF, cfg.MODEL.NETD,
                                            cfg.MODEL.N_LAYERS_D, norm, cfg.MODEL.INIT_TYPE, cfg.MODEL.INIT_GAIN,
                                            self.gpu_ids)
            # self.netD3 = networks.define_D(input_nc_D, cfg.MODEL.NDF, cfg.MODEL.NETD,
            #                                 cfg.MODEL.N_LAYERS_D, norm, cfg.MODEL.INIT_TYPE, cfg.MODEL.INIT_GAIN,
                                            # self.gpu_ids)
    def forward(self, input_rgb, img_rgb_M, input_ir, img_ir_M, testmode=0):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real_rgb_O = input_rgb.to(cfg.MODEL.DEVICE)
        self.real_rgb_M = img_rgb_M.to(cfg.MODEL.DEVICE)
        self.real_ir_O = input_ir.to(cfg.MODEL.DEVICE)
        self.real_ir_M = img_ir_M.to(cfg.MODEL.DEVICE)
        # for test
        if (not self.isTrain):
            if testmode==0:
                self.fake_P = input_rgb
                self.fake_P = self.netG(input_rgb, input_ir, mode = 'rgb')
                _,feat_OI = self.netC(input_rgb, input_ir, 1)
                _,feat_PI = self.netC(self.fake_P, self.fake_P, 1)

                feat_OI.to(cfg.MODEL.DEVICE)
                feat_PI.to(cfg.MODEL.DEVICE)
            else:
                self.fake_P = input_ir
                self.fake_P = self.netG(input_rgb, input_ir, mode = 'ir')

                _,feat_OI = self.netC(input_rgb, input_ir, 2)
                _,feat_PI = self.netC(self.fake_P, self.fake_P, 2)  
                feat_OI.to(cfg.MODEL.DEVICE)
                feat_PI.to(cfg.MODEL.DEVICE)
            
            return feat_OI, feat_PI
        # for train
           
        self.fake_rgb_P, self.fake_ir_P = self.netG(self.real_rgb_O, self.real_ir_O)

        if cfg.MODEL.POSE_GUIDE == 'on':
            self.mask_rgb = self.mask_head_rgb(self.fake_rgb_P)
            self.mask_ir = self.mask_head_ir(self.fake_ir_P)

        
        self.pred_OI, self.global_feat_OI = self.netC(self.real_rgb_O, self.real_ir_O, 0) 
        self.pred_OI.to(cfg.MODEL.DEVICE)
        self.global_feat_OI.to(cfg.MODEL.DEVICE)
        # self.feat_O_rgb.to(cfg.MODEL.DEVICE)
        # self.feat_O_ir.to(cfg.MODEL.DEVICE)
        
        
        if self.output_mode == 'protected':
            self.pred_PI, self.global_feat_PI = self.netC(self.fake_rgb_P, self.fake_ir_P, 0)
            self.pred_PI.to(cfg.MODEL.DEVICE)
            self.global_feat_PI.to(cfg.MODEL.DEVICE)
            # self.feat_P_rgb.to(cfg.MODEL.DEVICE)
            # self.feat_P_ir.to(cfg.MODEL.DEVICE)
            
            return self.pred_PI, self.global_feat_PI
        elif self.output_mode == 'both':
            self.pred_PI, self.global_feat_PI = self.netC(self.fake_rgb_P, self.fake_ir_P, 0)
            self.pred_PI.to(cfg.MODEL.DEVICE)
            self.global_feat_PI.to(cfg.MODEL.DEVICE)
            # self.feat_P_rgb.to(cfg.MODEL.DEVICE)
            # self.feat_P_ir.to(cfg.MODEL.DEVICE)
            return self.pred_OI, self.global_feat_OI, self.pred_PI, self.global_feat_PI
        else:
            return self.pred_OI, self.global_feat_OI


    def get_optimizer(self, cfg, criterion):
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        optimizer = {}
        
        ignored_params = list(map(id, self.netC.module.bottleneck.parameters())) \
                    + list(map(id, self.netC.module.classifier.parameters()))
        # ignored_params = list(map(id, self.netC.module.bottleneck.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self.netC.module.parameters())
        
        params_g = [{"params": self.netG.parameters(), "lr": lr, "weight_decay": weight_decay},
                    {"params": self.mask_head_rgb.parameters(), "lr": lr, "weight_decay": weight_decay} ,
                    {"params": self.mask_head_ir.parameters(), "lr": lr, "weight_decay": weight_decay}
        ]
        
        # params_c = [{"params": self.netC.parameters(), "lr": lr, "weight_decay": weight_decay}]
        
        params_c = [{'params': base_params, 'lr': 0.1 * 0.1},
                    {'params': self.netC.module.bottleneck.parameters(), 'lr': 0.1},
                    {'params': self.netC.module.classifier.parameters(), 'lr': 0.1}]
        # params_c_p = [{'params': base_params, 'lr': 0},
        #             {'params': self.netC.module.bottleneck.parameters(), 'lr': 0},
        #             {'params': self.netC.module.classifier.parameters(), 'lr': 0}]           
        params_d1 = [{"params": self.netD1.parameters(), "lr": lr, "weight_decay": weight_decay}]
        params_d2 = [{"params": self.netD2.parameters(), "lr": lr, "weight_decay": weight_decay}]
        # params_d3 = [{"params": self.netD3.parameters(), "lr": lr, "weight_decay": weight_decay}]
        if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
            optimizer['g'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_g, momentum=cfg.SOLVER.MOMENTUM)
            optimizer['c'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_c, weight_decay= weight_decay, momentum=cfg.SOLVER.MOMENTUM, nesterov=True)
            optimizer['d1'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_d1,
                                                                                momentum=cfg.SOLVER.MOMENTUM)
            optimizer['d2'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_d2,
                                                                                momentum=cfg.SOLVER.MOMENTUM)
            # optimizer['d3'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_d3,
            #                                                                     momentum=cfg.SOLVER.MOMENTUM)
        else:
            optimizer['g'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_g, betas=(0.5, 0.999))
            optimizer['c'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_c)
            optimizer['d1'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_d1, betas=(0.5, 0.999))
            optimizer['d2'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_d2, betas=(0.5, 0.999))
            # optimizer['d3'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params_d3, betas=(0.5, 0.999))
            
        if cfg.MODEL.CENTER_LOSS == 'on':
            optimizer['center'] = torch.optim.SGD(criterion['center'].parameters(), lr=cfg.SOLVER.CENTER_LR)

        return optimizer

    def get_creterion(self, cfg, num_classes):
        criterion = {}
        # criterion['xent'] = CrossEntropyLabelSmooth(num_classes=num_classes).to(cfg.MODEL.DEVICE)  # new add by luo
        criterion['xent'] = nn.CrossEntropyLoss().to(cfg.MODEL.DEVICE)
        print("Weighted Regularized Triplet:", cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET)
        if cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET == 'on':
            criterion['triplet'] = TripletLoss_WRT()
        else:
            criterion['triplet'] = TripletLoss_ADP() 

        if cfg.MODEL.CENTER_LOSS == 'on':
            criterion['center'] = CenterLoss(num_classes=num_classes, feat_dim=cfg.MODEL.CENTER_FEAT_DIM,
                                             use_gpu=True)
        
        criterion['anonloss'] = TwoStageAnonymizationLoss(num_classes=num_classes, feat_dim=cfg.MODEL.CENTER_FEAT_DIM).to(cfg.MODEL.DEVICE)
        criterion['comploss'] = TwoStageAnonymizationLoss(num_classes=num_classes, feat_dim=cfg.MODEL.CENTER_FEAT_DIM).to(cfg.MODEL.DEVICE)
        
        def criterion_reid(score_O, feat_O,score_P, feat_P, targets):
            
            xent_loss_O = criterion['xent'](score_O, targets.long())
            xent_loss_P = criterion['xent'](score_P, targets.long())
            loss = 0 
            loss =  xent_loss_O + xent_loss_P
            if cfg.MODEL.FUSION=='on':
                f_feat = self.fusion(feat_O,feat_P)
                triplet_F_loss, correct_F = criterion['triplet'](f_feat, targets)
            else:
                triplet_F_loss, correct_F= criterion['triplet'](feat_O, targets)
            
            loss = loss + triplet_F_loss

            if cfg.MODEL.ANON_LOSS == 'on':
                anon_loss = criterion['anonloss'](feat_O, feat_P)
                loss = loss + anon_loss
           

            if cfg.MODEL.CENTER_LOSS == 'on':
                loss = loss + cfg.SOLVER.CENTER_LOSS_WEIGHT * criterion['center'](feat_O, targets) #+ cfg.SOLVER.CENTER_LOSS_WEIGHT * criterion['center'](feat_P, targets)
            # correct = (correct_O + correct_P) / 2
            return loss, correct_F
        
        def criterion_ts(t_f,s_f):
            dist_loss = F.mse_loss(t_f,s_f,reduction='sum')
            # def kl(x, y, T):
            #     pred_x = F.log_softmax(x/T,dim=1)
            #     target_y = F.softmax(y/T,dim=1)
            #     target_y =target_y+10**(-7)
            #     #kld=T*T*((target_y*(target_y.log()-pred_x)).sum(1).sum()/target_y.size()[0])
            #     kld=T*T*((target_y*(target_y.log()-pred_x)).sum(1).sum())
            #     return kld
            # dist_loss = kl(t_f,s_f,3)
            return dist_loss
        
        def _build_pose_conf_map(batch_images, min_conf, radius):
            """
            batch_images: [B,3,H,W] in [0,1]
            return: C_pose [B,1,H,W]
            """
            B, _, H, W = batch_images.shape
            device = batch_images.device
            C_pose = torch.zeros((B, 1, H, W), device=device, dtype=batch_images.dtype)

            # YOLOv8n-pose
            imgs_list = []
            for b in range(B):
                img = (batch_images[b].detach().clamp(0,1) * 255.0).to(torch.uint8).permute(1,2,0).cpu().numpy()  # HWC uint8
                imgs_list.append(img)
            imgsz = cfg.MODEL.POSE_SHORT_SIDE
            with torch.inference_mode():
                # results = self.pose_model(imgs_list, imgsz=imgsz, conf=float(min_conf), verbose=False, device=self.cfg.MODEL.DEVICE)
                results = self.pose_model(imgs_list, imgsz=imgsz, conf=float(min_conf), verbose=False, device='cpu')

            yy, xx = torch.meshgrid(torch.arange(H, device=device),
                                    torch.arange(W, device=device), indexing='ij')
            for b in range(B):
                res = results[b]
                if res is None or res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
                    continue
                # Select the human instance with the highest confidence
                confs = res.boxes.conf.cpu().numpy()
                best = int(confs.argmax())
                # Key point relative coordinates [K,2], range [0,1]
                kps = res.keypoints.xyn[best].cpu().numpy()  # (K,2)
                # If keypoint confidence is available, filter out low-confidence keypoints
                kps_conf = None
                if hasattr(res.keypoints, 'conf') and res.keypoints.conf is not None:
                    kps_conf = res.keypoints.conf[best].cpu().numpy()  # (K,)
                for k in range(kps.shape[0]):
                    if kps_conf is not None and kps_conf[k] < float(min_conf):
                        continue
                    cx = float(kps[k,0]) * W
                    cy = float(kps[k,1]) * H
                    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
                    C_pose[b, 0] = torch.maximum(C_pose[b, 0], (dist2 <= (radius * radius)).float())
            return C_pose.clamp_(0, 1)

        def criterion_G_guidance():
            mean = torch.tensor(self.cfg.INPUT.PIXEL_MEAN, device=self.real_rgb_O.device).view(1,3,1,1)
            std  = torch.tensor(self.cfg.INPUT.PIXEL_STD,  device=self.real_rgb_O.device).view(1,3,1,1)
            rgb_01 = (self.real_rgb_O * std + mean).clamp(0, 1)  # 还原到[0,1]
            ir_01 = (self.real_ir_O * std + mean).clamp(0, 1)  # 还原到[0,1]


            C_pose_rgb = _build_pose_conf_map(rgb_01, self.cfg.MODEL.POSE_MIN_CONF, int(self.cfg.MODEL.POSE_RADIUS))
            C_pose_ir = _build_pose_conf_map(ir_01, self.cfg.MODEL.POSE_MIN_CONF, int(self.cfg.MODEL.POSE_RADIUS))
            M_unet_rgb = self.mask_rgb
            M_unet_ir = self.mask_ir

            # loss_guidance = torch.mean(M_unet * (1.0 - C_pose)) + torch.mean((1.0 - M_unet) * C_pose)
            # loss_guidance = torch.mean(M_unet * C_pose) + torch.mean((1.0 - M_unet) * (1.0 - C_pose))
            loss_guidance = torch.mean((M_unet_rgb - C_pose_rgb) ** 2) + torch.mean((M_unet_ir - C_pose_ir) ** 2)
            return loss_guidance
    
        def criterion_dcma():
            """
            Compute the consistency of the anonymized difference vector (f_raw - f_anno) between RGB and IR modalities
            """
            b_rgb = self.real_rgb_O.size(0)
            b_ir  = self.real_ir_O.size(0)
            b_pair = min(b_rgb, b_ir)

            # Extract modality features from the fused features
            # self.global_feat_OI is the concatenation of  [f_rgb_raw, f_ir_raw] 
            # self.global_feat_PI is the concatenation of  [f_rgb_anno, f_ir_anno]
            f_OI = self.global_feat_OI
            f_PI = self.global_feat_PI

            f_rgb_raw  = f_OI[:b_rgb][:b_pair]
            f_ir_raw   = f_OI[b_rgb : b_rgb + b_ir][:b_pair]

            f_rgb_anno = f_PI[:b_rgb][:b_pair]
            f_ir_anno  = f_PI[b_rgb : b_rgb + b_ir][:b_pair]

            gap_rgb = f_rgb_raw - f_rgb_anno
            gap_ir  = f_ir_raw - f_ir_anno

            loss_dcma = F.mse_loss(gap_rgb, gap_ir)
            
            return loss_dcma

        
        criterion['reid'] = criterion_reid
        # criterion['anon'] = criterion_anon
        criterion['ts'] = criterion_ts
        criterion['gan'] = networks.GANLoss('vanilla').to(cfg.MODEL.DEVICE)
        criterion['l1'] = torch.nn.L1Loss()
        criterion['G_guidance'] = criterion_G_guidance
        criterion['dcma'] = criterion_dcma


        def criterion_G_gan():
            fake_rgb_OP = torch.cat((self.real_rgb_O, self.fake_rgb_P), 1)
            fake_ir_OP = torch.cat((self.real_ir_O, self.fake_ir_P), 1)
            pred_fake_rgb_OP = self.netD1(fake_rgb_OP)
            if cfg.MODEL.SCHEMA == "dual":
                pred_fake_ir_OP = self.netD2(fake_ir_OP)
            else:
                pred_fake_ir_OP = self.netD1(fake_ir_OP)
            
            
            # rgb_O_ir_P = torch.cat((self.real_rgb_O, torch.cat((self.fake_ir_P, self.fake_ir_P),0)), 1)
            # ir_O_rgb_P = torch.cat((torch.cat((self.real_ir_O, self.real_ir_O),0), self.fake_rgb_P), 1)
            
            # pred_rgb_O_ir_P = self.netD3(rgb_O_ir_P)
            # pred_ir_O_rgb_P = self.netD3(ir_O_rgb_P)
            
            # loss_g_gan = criterion['gan'](pred_fake_rgb_OP, True) + criterion['gan'](pred_fake_ir_OP, True) + criterion['gan'](pred_rgb_O_ir_P, True) + criterion['gan'](pred_ir_O_rgb_P, True)
            loss_g_gan = criterion['gan'](pred_fake_rgb_OP, True) + criterion['gan'](pred_fake_ir_OP, True)
            return loss_g_gan

        def criterion_G_l1():
            loss_g_rgb_l1 = torch.nn.L1Loss()(self.fake_rgb_P, self.real_rgb_M)
            loss_g_ir_l1 = torch.nn.L1Loss()(self.fake_ir_P, self.real_ir_M)
            loss_g_l1=loss_g_rgb_l1 + loss_g_ir_l1
            return loss_g_l1


        def criterion_D1():
            # Fake; stop backprop to the generator by detaching fake_B
            fake_rgb_OP = torch.cat((self.real_rgb_O, self.fake_rgb_P),
                                1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake_rgb = self.netD1(fake_rgb_OP.detach())
            loss_d_fake_rgb = criterion['gan'](pred_fake_rgb, False)
            # Real
            real_rgb_OM = torch.cat((self.real_rgb_O, self.real_rgb_M), 1)
            pred_real_rgb = self.netD1(real_rgb_OM.detach())
            loss_d_real_rgb = criterion['gan'](pred_real_rgb, True)
            # combine loss and calculate gradients
            loss = loss_d_fake_rgb + loss_d_real_rgb
            return loss
        
        def criterion_D2():
            # Fake; stop backprop to the generator by detaching fake_B
            fake_ir_OP = torch.cat((self.real_ir_O, self.fake_ir_P),
                                1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            if cfg.MODEL.SCHEMA == "dual":
                pred_fake_ir = self.netD2(fake_ir_OP.detach())
            else:
                pred_fake_ir = self.netD1(fake_ir_OP.detach())
                
            loss_d_fake_ir = criterion['gan'](pred_fake_ir, False)
            # Real
            real_ir_OM = torch.cat((self.real_ir_O, self.real_ir_M), 1)
            if cfg.MODEL.SCHEMA == "dual":
                pred_real_ir = self.netD2(real_ir_OM.detach())
            else:
                pred_real_ir = self.netD1(real_ir_OM.detach())
            loss_d_real_ir = criterion['gan'](pred_real_ir, True)
            # combine loss and calculate gradients
            loss = loss_d_fake_ir + loss_d_real_ir
            return loss
        
        # def criterion_D3():
        #     # Fake; stop backprop to the generator by detaching fake_B
        #     rgb_O_ir_P = torch.cat((self.real_rgb_O, torch.cat((self.fake_ir_P, self.fake_ir_P),0)),
        #                         1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        #     pred_rgb_O_ir_P = self.netD3(rgb_O_ir_P.detach())
        #     loss_d_rgb_O_ir_P = criterion['gan'](pred_rgb_O_ir_P, False)
            
        #     # Fake; stop backprop to the generator by detaching fake_B
        #     ir_O_rgb_P = torch.cat((torch.cat((self.real_ir_O, self.real_ir_O),0), self.fake_rgb_P),
        #                         1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        #     pred_ir_O_rgb_P = self.netD3(ir_O_rgb_P.detach())
        #     loss_d_ir_O_rgb_P = criterion['gan'](pred_ir_O_rgb_P, False)
            
        #     # Real
        #     rgb_O_ir_M = torch.cat((self.real_rgb_O, torch.cat((self.real_ir_M, self.real_ir_M),0)), 1)
        #     pred_rgb_O_ir_M = self.netD3(rgb_O_ir_M.detach())
        #     loss_d_rgb_O_ir_M = criterion['gan'](pred_rgb_O_ir_M, True)
            
        #     # Real
        #     ir_O_rgb_M = torch.cat((torch.cat((self.real_ir_O, self.real_ir_O),0), self.real_rgb_M), 1)
        #     pred_ir_O_rgb_M = self.netD3(ir_O_rgb_M.detach())
        #     loss_d_ir_O_rgb_M = criterion['gan'](pred_ir_O_rgb_M, True)
        #     # combine loss and calculate gradients
        #     loss = loss_d_rgb_O_ir_P + loss_d_ir_O_rgb_P + loss_d_rgb_O_ir_M + loss_d_ir_O_rgb_M
        #     return loss
        

        criterion['G_gan'] = criterion_G_gan
        criterion['G_l1'] = criterion_G_l1
        criterion['D1'] = criterion_D1
        criterion['D2'] = criterion_D2
        # criterion['D3'] = criterion_D3

        return criterion

    def load_param(self, cfg):
        if isinstance(cfg.MODEL.EPOCH, int):
            epoch = 'epoch_%d' % cfg.MODEL.EPOCH
        else:
            epoch = 'epoch_' + cfg.MODEL.EPOCH
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(cfg.MODEL.PRETRAIN_DIR, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    # if name != 'C':
                    #     net = net.module
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(torch.device(cfg.MODEL.DEVICE)))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                if name == 'C':
                    if not isinstance(state_dict, collections.OrderedDict):
                        state_dict = state_dict.state_dict()
                    for i in state_dict:
                        if 'classifier' in i:
                            continue
                        net.state_dict()[i].copy_(state_dict[i])
                else:
                    # patch InstanceNorm checkpoints prior to 0.4
                    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                        self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                    net.load_state_dict(state_dict)

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    if name == "C":
                        torch.save(net.module.state_dict(), save_path)
                        # net.cuda()
                        net.to(cfg.MODEL.DEVICE)
                    else:
                        torch.save(net.module.cpu().state_dict(), save_path)
                        # net.cuda()
                        net.to(cfg.MODEL.DEVICE)
                else:
                    if name == "C":
                        torch.save(net.module.cpu().state_dict(), save_path)
                    else:
                        torch.save(net.module.cpu().state_dict(), save_path)

    def eval(self):
        self.isTrain = False
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    # modified HERE
    def train(self):
        """Make models eval mode during test time"""
        self.isTrain = True
        for name in self.model_names:
            # Here
            # if name == 'C':
            #     continue

            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
