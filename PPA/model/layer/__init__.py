# encoding: utf-8

from .center_loss import CenterLoss
from .kl_loss import KLDivLoss
from .triplet_loss import CrossEntropyLabelSmooth, TripletLoss, WeightedRegularizedTriplet, TripletLoss_ADP, TripletLoss_WRT, TwoStageAnonymizationLoss
from .non_local import Non_local
from .gem_pool import GeneralizedMeanPooling, GeneralizedMeanPoolingP