# encoding: utf-8
import torch
from torch import nn
import torch.nn.functional as F
from config import cfg

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(
            dist_mat, labels)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        # print(log_probs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)

        # if self.use_gpu: targets = targets.cuda()
        if self.use_gpu: targets = targets.to(cfg.MODEL.DEVICE)

        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


class WeightedRegularizedTriplet(object):

    def __init__(self):
        self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()).float()
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        return loss, furthest_positive, closest_negative

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    # dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx + (-2 * emb1.mm(emb2.t()))
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx 


class TripletLoss_ADP(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self, alpha =1, gamma = 1, square = 1):
        super(TripletLoss_ADP, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()
        self.alpha = alpha
        self.gamma = gamma
        self.square = square

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap*self.alpha, is_pos)
        weights_an = softmax_weights(-dist_an*self.alpha, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)
        # print('furthest_positive', furthest_positive)
        # print('closest_negative', closest_negative)
        
        # ranking_loss = nn.SoftMarginLoss(reduction = 'none')
        # loss1 = ranking_loss(closest_negative - furthest_positive, y)
        
        # squared difference
        if self.square ==0:
            y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
            loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)
        else:
            diff_pow = torch.pow(furthest_positive - closest_negative, 2) * self.gamma

            diff_pow =torch.clamp_max(diff_pow, max=88)

            # Compute ranking hinge loss
            y1 = (furthest_positive > closest_negative).float()
            y2 = y1 - 1
            y = -(y1 + y2)
            
            loss = self.ranking_loss(diff_pow, y)
        
        # loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct
    
class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative  = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct
    


# class TwoStageAnonymizationLoss(nn.Module):
#     def __init__(self, num_classes, feat_dim, epsilon=1e-8):
#         super().__init__()
#         self.num_classes = num_classes
#         self.feat_dim = feat_dim
#         self.epsilon = epsilon

#         # 初始化可学习的质心
#         self.visible_centroids = nn.Parameter(torch.randn(num_classes, feat_dim))
#         self.infrared_centroids = nn.Parameter(torch.randn(num_classes, feat_dim))

#     def _safe_softmax(self, x, dim):
#         """防止softmax输出为0的softmax"""
#         probs = F.softmax(x, dim=dim)
#         # 确保prob在计算时不会为0
#         probs = (probs + self.epsilon) / (1 + self.epsilon * probs.size(dim))
#         return probs

#     def _safe_log(self, x):
#         """防止log(0)产生nan"""
#         return torch.log(x + self.epsilon)

#     def forward(self, feat_O, feat_P, targets_rgb, targets_ir):
#         n1 = feat_O.shape[0] // 3        
#         visible_feat = feat_O.narrow(0, 0, n1)
#         infrared_feat = feat_O.narrow(0, 2 * n1, n1)
        
#         n2 = feat_P.shape[0] // 3        
#         visible_anon_feat = feat_P.narrow(0, 0, n2)
#         infrared_anon_feat = feat_P.narrow(0, 2 * n2, n2)
        
#         visible_labels = targets_rgb
#         infrared_labels = targets_ir

#         # 计算可见光与匿名可见光的KL散度
#         visible_log = self._safe_log(F.softmax(visible_feat, dim=1))
#         visible_anon_prob = self._safe_softmax(visible_anon_feat, dim=1)
#         kl_visible = F.kl_div(visible_log, visible_anon_prob, reduction='batchmean')

#         # 计算红外与匿名红外的KL散度
#         infrared_log = self._safe_log(F.softmax(infrared_feat, dim=1))
#         infrared_anon_prob = self._safe_softmax(infrared_anon_feat, dim=1)
#         kl_infrared = F.kl_div(infrared_log, infrared_anon_prob, reduction='batchmean')

#         # 获取可见光特征对应的质心
#         vis_centroids = self.visible_centroids[visible_labels]
#         vis_centroid_prob = self._safe_softmax(vis_centroids, dim=1)
#         kl_vis_centroid = F.kl_div(visible_log, vis_centroid_prob, reduction='batchmean')

#         # 获取红外特征对应的质心
#         ir_centroids = self.infrared_centroids[infrared_labels]
#         ir_centroid_prob = self._safe_softmax(ir_centroids, dim=1)
#         kl_ir_centroid = F.kl_div(infrared_log, ir_centroid_prob, reduction='batchmean')

#         # 计算质心间的对称KL散度
#         # 可见质心到红外质心
#         vis_centroid_log = self._safe_log(F.softmax(self.visible_centroids, dim=1))
#         ir_centroid_prob = self._safe_softmax(self.infrared_centroids, dim=1)
#         kl_vis2ir = F.kl_div(vis_centroid_log, ir_centroid_prob, reduction='sum')

#         # 红外质心到可见质心
#         ir_centroid_log = self._safe_log(F.softmax(self.infrared_centroids, dim=1))
#         vis_centroid_prob = self._safe_softmax(self.visible_centroids, dim=1)
#         kl_ir2vis = F.kl_div(ir_centroid_log, vis_centroid_prob, reduction='sum')

#         # 总质心损失
#         kl_centroid = (kl_vis2ir + kl_ir2vis) / self.num_classes
#         # print("kl_visible",kl_visible)
#         # print("kl_infrared",kl_infrared)
#         # print("kl_vis_centroid",kl_vis_centroid)
#         # print("kl_ir_centroid",kl_ir_centroid)
#         # print("kl_centroid",kl_centroid)
#         # 总损失
#         total_loss = 100 * kl_visible + 100 * kl_infrared + kl_vis_centroid + kl_ir_centroid + kl_centroid

#         return total_loss


class TwoStageAnonymizationLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, epsilon=1e-8, T = 2):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.epsilon = epsilon
        self.T = T



    def kl(self, x, y, T):
        pred_x = F.log_softmax(x/T,dim=1)
        target_y = F.softmax(y/T,dim=1)
        target_y =target_y+10**(-7)
        kld=T*T*((target_y*(target_y.log()-pred_x)).sum(1).sum()/target_y.size()[0])
        return kld



    def forward(self, feat_O, feat_P):
        # n1 = feat_O.shape[0] // 3        
        # visible_feat = feat_O.narrow(0, 0, n1)
        # infrared_feat = feat_O.narrow(0, 2 * n1, n1)
        
        # n2 = feat_P.shape[0] // 3        
        # visible_anon_feat = feat_P.narrow(0, 0, n2)
        # infrared_anon_feat = feat_P.narrow(0, 2 * n2, n2)
        
        # # n3 = inter_feat_O.shape[0] // 3        
        # # visible_inter_feat = inter_feat_O.narrow(0, 0, n3)
        # # infrared_inter_feat = inter_feat_O.narrow(0, 2 * n3, n3)
        
        # # n4 = inter_feat_P.shape[0] // 3        
        # # visible_inter_anon_feat = inter_feat_P.narrow(0, 0, n4)
        # # infrared_inter_anon_feat = inter_feat_P.narrow(0, 2 * n4, n4)
        
        # # visible_labels = targets_rgb
        # # infrared_labels = targets_ir

        # kl_visible = self.kl(visible_feat, visible_anon_feat, self.T)

        # kl_infrared = self.kl(infrared_feat, infrared_anon_feat, self.T)
        
        # # inter_visible = nn.MSELoss()(visible_inter_anon_feat, visible_inter_feat)
        # # print(inter_visible)

        # # inter_infrared = nn.MSELoss()(infrared_inter_anon_feat, infrared_inter_feat)

        
        # total_loss = kl_visible + kl_infrared
        
        total_loss =F.mse_loss(feat_O, feat_P)

        return total_loss

