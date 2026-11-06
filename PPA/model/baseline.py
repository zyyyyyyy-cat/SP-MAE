# encoding: utf-8

import os
import torch
from torch import nn
from torch.nn import init
from functools import partial

# from timm.models.layers import trunc_normal_tf_
# from timm.models.helpers import named_apply
from .layer import CrossEntropyLabelSmooth, TripletLoss, WeightedRegularizedTriplet, CenterLoss, GeneralizedMeanPooling, \
    GeneralizedMeanPoolingP

from config import cfg
import torch.utils.model_zoo as model_zoo
import math
from .backbones.resnet_nl import ResNetNL

class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    # original padding is 1; original dilation is 1
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):

  def __init__(self, block, layers, last_conv_stride=2, last_conv_dilation=1):

    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                           bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=last_conv_stride, dilation=last_conv_dilation)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, dilation))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    return x

model_urls = {
  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
  'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
  'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def remove_fc(state_dict):
  """Remove the fc layer parameters from state_dict."""
  # for key, value in state_dict.items():
  for key, value in list(state_dict.items()):
    if key.startswith('fc.'):
      del state_dict[key]
  return state_dict

def resnet50(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
  if pretrained:
    model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])))

  return model


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)



        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z



def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


# def _init_weights(module, name, scheme=''):
#     if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
#         if scheme == 'normal':
#             nn.init.normal_(module.weight, std=.02)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#         elif scheme == 'trunc_normal':
#             trunc_normal_tf_(module.weight, std=.02)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#         elif scheme == 'xavier_normal':
#             nn.init.xavier_normal_(module.weight)
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#         elif scheme == 'kaiming_normal':
#             nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#         else:
#             # efficientnet like
#             fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
#             fan_out //= module.groups
#             nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
#             if module.bias is not None:
#                 nn.init.zeros_(module.bias)
#     elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
#         nn.init.constant_(module.weight, 1)
#         nn.init.constant_(module.bias, 0)
#     elif isinstance(module, nn.LayerNorm):
#         nn.init.constant_(module.weight, 1)
#         nn.init.constant_(module.bias, 0)

# def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
#     # activation layer
#     act = act.lower()
#     if act == 'relu':
#         layer = nn.ReLU(inplace)
#     elif act == 'relu6':
#         layer = nn.ReLU6(inplace)
#     elif act == 'leakyrelu':
#         layer = nn.LeakyReLU(neg_slope, inplace)
#     elif act == 'prelu':
#         layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
#     elif act == 'gelu':
#         layer = nn.GELU()
#     elif act == 'hswish':
#         layer = nn.Hardswish(inplace)
#     else:
#         raise NotImplementedError('activation layer [%s] is not found' % act)
#     return layer


# def channel_shuffle(x, groups):
#     batchsize, num_channels, height, width = x.data.size()
#     channels_per_group = num_channels // groups    
#     # reshape
#     x = x.view(batchsize, groups, 
#                channels_per_group, height, width)
#     x = torch.transpose(x, 1, 2).contiguous()
#     # flatten
#     x = x.view(batchsize, -1, height, width)
#     return x

# class EUCB(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
#         super(EUCB,self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.up_dwc = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=self.in_channels, bias=False),
# 	        nn.BatchNorm2d(self.in_channels),
#             act_layer(activation, inplace=True)
#         )
#         self.pwc = nn.Sequential(
#             nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
#         ) 
#         self.init_weights('normal')
    
#     def init_weights(self, scheme=''):
#         named_apply(partial(_init_weights, scheme=scheme), self)

#     def forward(self, x):
#         x = self.up_dwc(x)
#         x = channel_shuffle(x, self.in_channels)
#         x = self.pwc(x)
#         return x





class visible_module(nn.Module):
    def __init__(self,cfg):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v
        # self.visible.to(cfg.MODEL.DEVICE)
    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, cfg):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t
        # self.thermal.to(cfg.MODEL.DEVICE)

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x

class base_resnet(nn.Module):
    def __init__(self, cfg):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
        # self.base.to(cfg.MODEL.DEVICE)
        
    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class Baseline(nn.Module):

    def __init__(self, num_classes, gem_pool):
        super(Baseline, self).__init__()
        
        self.thermal_module = thermal_module(cfg)
        self.visible_module = visible_module(cfg)
        self.base_resnet = base_resnet(cfg)
        # self.ecub_rgb=EUCB(in_channels=2048, out_channels=64, kernel_size=3, stride=1)
        layers=[3, 4, 6, 3]
        non_layers=[0,2,3,0]
        self.NL_1 = nn.ModuleList(
            [Non_local(256) for i in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512) for i in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024) for i in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048) for i in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        pool_dim = 2048
        self.num_classes = num_classes
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gem_pool


    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1) #[64, 64, 64, 32]
            x2 = self.thermal_module(x2) #[32, 64, 64, 32]
            x = torch.cat((x1, x2), 0)
            
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared block

        NL1_counter = 0
        if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
        for i in range(len(self.base_resnet.base.layer1)):
            x = self.base_resnet.base.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
        
        
        # Layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
        for i in range(len(self.base_resnet.base.layer2)):
            x = self.base_resnet.base.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
        
        
        # Layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
        for i in range(len(self.base_resnet.base.layer3)):
            x = self.base_resnet.base.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
                
        
        # Layer 4
        NL4_counter = 0
        if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
        for i in range(len(self.base_resnet.base.layer4)):
            x = self.base_resnet.base.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1
                
        # xup=x
        # xup=self.ecub_rgb(xup)
        # print(xup.shape)
                
        if self.gm_pool  == 'on':
            b, c, h, w = x.shape #[96, 2048, 16, 8]
            
            x = x.view(b, c, -1) #[96, 2048, 128]
            p = 3.0
            x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p) #[96, 2048]
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        feat = self.bottleneck(x_pool)

        if self.training:
            return self.classifier(feat), x_pool
        else:
            return self.l2norm(feat), self.l2norm(x_pool)
        # if self.training:
        #     return x_pool, self.classifier(feat)
        # else:
        #     return self.l2norm(x_pool), self.l2norm(feat)



    # def get_optimizer(self, cfg, criterion):
    #     optimizer = {}
    #     params = []
    #     lr = cfg.SOLVER.BASE_LR
    #     weight_decay = cfg.SOLVER.WEIGHT_DECAY
    #     for key, value in self.named_parameters():
    #         if not value.requires_grad:
    #             continue
    #         params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    #     if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
    #         optimizer['model'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    #     else:
    #         optimizer['model'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    #     if cfg.MODEL.CENTER_LOSS == 'on':
    #         optimizer['center'] = torch.optim.SGD(criterion['center'].parameters(), lr=cfg.SOLVER.CENTER_LR)
    #     return optimizer

    # def get_creterion(self, cfg, num_classes):
    #     criterion = {}
    #     criterion['xent'] = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo

    #     print("Weighted Regularized Triplet:", cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET)
    #     if cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET == 'on':
    #         criterion['triplet'] = WeightedRegularizedTriplet()
    #     else:
    #         criterion['triplet'] = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss

    #     if cfg.MODEL.CENTER_LOSS == 'on':
    #         criterion['center'] = CenterLoss(num_classes=num_classes, feat_dim=cfg.MODEL.CENTER_FEAT_DIM,
    #                                          use_gpu=True)

    #     def criterion_total(score, feat, target):
    #         loss = criterion['xent'](score, target) + criterion['triplet'](feat, target)[0]
    #         if cfg.MODEL.CENTER_LOSS == 'on':
    #             loss = loss + cfg.SOLVER.CENTER_LOSS_WEIGHT * criterion['center'](feat, target)
    #         return loss

    #     criterion['total'] = criterion_total

    #     return criterion

    # def load_param(self, cfg, net):
    #     load_filename = 'teacher_model.t'
    #     load_path = os.path.join(cfg.PRETRAINED_TEACHER, load_filename)
    #     print('==> loading checkpoint {}'.format(load_path))
    #     checkpoint = torch.load(load_path)
    #     net.load_state_dict(checkpoint['net'])
    
class AGW(nn.Module):
    

    def __init__(self, num_classes, last_stride, model_path, model_name, gem_pool, pretrain_choice):
        super(AGW, self).__init__()
        self.base = ResNetNL(last_stride=last_stride,
                                 block=Bottleneck,
                                 layers=[3, 4, 6, 3],
                                 non_layers=[0, 2, 3, 0])
        in_planes = 2048        
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.num_classes = num_classes

        if gem_pool == 'on':
            print("Generalized Mean Pooling")
            self.global_pool = GeneralizedMeanPoolingP()
        else:
            print("Global Adaptive Pooling")
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.bottleneck = nn.BatchNorm1d(in_planes)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(in_planes, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = x1
        elif modal == 2:
            x = x2
            
            
        x = self.base(x)
        global_feat = self.global_pool(x)  # (b, 2048, 1, 1)
        
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if not self.training:
            return feat,feat

        cls_score = self.classifier(feat)
        return cls_score, global_feat


    # def load_param(self, trained_path):
    #     param_dict = torch.load(trained_path)
    #     if not isinstance(param_dict, collections.OrderedDict):
    #         param_dict = param_dict.state_dict()
    #     for i in param_dict:
    #         if 'classifier' in i:
    #             continue
    #         self.state_dict()[i].copy_(param_dict[i])

    def get_optimizer(self, cfg, criterion):
        optimizer = {}
        params = []
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        for key, value in self.named_parameters():
            if not value.requires_grad:
                continue
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
            optimizer['model'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
        else:
            optimizer['model'] = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
        if cfg.MODEL.CENTER_LOSS == 'on':
            optimizer['center'] = torch.optim.SGD(criterion['center'].parameters(), lr=cfg.SOLVER.CENTER_LR)
        return optimizer

    def get_creterion(self, cfg, num_classes):
        criterion = {}
        criterion['xent'] = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo

        print("Weighted Regularized Triplet:", cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET)
        if cfg.MODEL.WEIGHT_REGULARIZED_TRIPLET == 'on':
            criterion['triplet'] = WeightedRegularizedTriplet()
        else:
            criterion['triplet'] = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss

        if cfg.MODEL.CENTER_LOSS == 'on':
            criterion['center'] = CenterLoss(num_classes=num_classes, feat_dim=cfg.MODEL.CENTER_FEAT_DIM,
                                             use_gpu=True)

        def criterion_total(score, feat, target):
            loss = criterion['xent'](score, target) + criterion['triplet'](feat, target)[0]
            if cfg.MODEL.CENTER_LOSS == 'on':
                loss = loss + cfg.SOLVER.CENTER_LOSS_WEIGHT * criterion['center'](feat, target)
            return loss

        criterion['total'] = criterion_total

        return criterion
    
    
class Teacher(nn.Module):

    def __init__(self, num_classes, gem_pool):
        super(Teacher, self).__init__()
        
        self.thermal_module = thermal_module(cfg)
        self.visible_module = visible_module(cfg)
        self.base_resnet = base_resnet(cfg)
        layers=[3, 4, 6, 3]
        non_layers=[0,2,3,0]
        self.NL_1 = nn.ModuleList(
            [Non_local(256) for i in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512) for i in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024) for i in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048) for i in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        pool_dim = 2048
        self.num_classes = num_classes
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gem_pool


    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1) #[64, 64, 64, 32]
            x2 = self.thermal_module(x2) #[32, 64, 64, 32]
            x = torch.cat((x1, x2), 0)
            
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared block

        NL1_counter = 0
        if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
        for i in range(len(self.base_resnet.base.layer1)):
            x = self.base_resnet.base.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
        
        
        # Layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
        for i in range(len(self.base_resnet.base.layer2)):
            x = self.base_resnet.base.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
        
        
        # Layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
        for i in range(len(self.base_resnet.base.layer3)):
            x = self.base_resnet.base.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
                
        
        # Layer 4
        NL4_counter = 0
        if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
        for i in range(len(self.base_resnet.base.layer4)):
            x = self.base_resnet.base.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1
                
                
        if self.gm_pool  == 'on':
            b, c, h, w = x.shape #[96, 2048, 16, 8]
            
            x = x.view(b, c, -1) #[96, 2048, 128]
            p = 3.0
            x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p) #[96, 2048]
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        feat = self.bottleneck(x_pool)

        if self.training:
            return self.classifier(feat), x_pool
        else:
            return self.classifier(feat), self.l2norm(x_pool)
        
    def load_param(self, cfg, net):
        if cfg.DATASETS.NAMES == 'sysu':
            load_filename = 'teacher_model_sysu.t'
        elif cfg.DATASETS.NAMES == 'regdb':
            # load_filename = 'teacher_model_regdb_trial_{}.t'.format(cfg.MODEL.TRIAL)
            load_filename = 'teacher_model_regdb.t'
        else:
            load_filename=''
        load_path = os.path.join(cfg.PRETRAINED_TEACHER, load_filename)
        print('==> loading checkpoint {}'.format(load_path))
        # 原来的代码（第761行）

        checkpoint = torch.load(load_path, map_location='cpu')
        # checkpoint = torch.load(load_path)
        net.load_state_dict(checkpoint['net'])
