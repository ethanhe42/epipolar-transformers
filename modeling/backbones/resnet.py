import logging
import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from modeling.layers.epipolar import Epipolar
from modeling import registry
from core import cfg
from .basic_batch import find_tensor_peak_batch
from utils.logger import setup_logger
from utils.model_serialization import load_state_dict

# logger = logging.getLogger(__name__)
logger = setup_logger("resnet", cfg.FOLDER_NAME)

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_channels = 512 * block.expansion
        #self.fc = nn.Linear(self.out_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

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

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.fc(x)

        return x


@registry.BACKBONES.register('R-18')
def resnet18(cfg, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if cfg.BACKBONE.PRETRAINED:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model


@registry.BACKBONES.register('R-34')
def resnet34(cfg, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if cfg.BACKBONE.PRETRAINED:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


@registry.BACKBONES.register('R-50')
def resnet50(cfg, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if cfg.BACKBONE.PRETRAINED:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


@registry.BACKBONES.register('R-101')
def resnet101(cfg, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if cfg.BACKBONE.PRETRAINED:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model


@registry.BACKBONES.register('R-152')
def resnet152(cfg, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if cfg.BACKBONE.PRETRAINED:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']), strict=False)
    return model


# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Chunyu Wang (chnuwa@microsoft.com), modified by Yihui He
# ------------------------------------------------------------------------------


class PoseResNet(nn.Module):

    def __init__(self, block, layers, cfg, **kwargs):
        if cfg.BACKBONE.BN_MOMENTUM < 0:
            self.BN_MOMENTUM = None
        else:
            self.BN_MOMENTUM = cfg.BACKBONE.BN_MOMENTUM

        DECONV_WITH_BIAS = False
        NUM_DECONV_LAYERS = 3
        NUM_DECONV_FILTERS = [256, 256, 256]
        NUM_DECONV_KERNELS = [4, 4, 4]
        FINAL_CONV_KERNEL = 1 #cfg.POSE_RESNET.FINAL_CONV_KERNEL
        self.inplanes = 64
        self.deconv_with_bias = DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=self.BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            NUM_DECONV_LAYERS,
            NUM_DECONV_FILTERS,
            NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=NUM_DECONV_FILTERS[-1],
            out_channels=cfg.KEYPOINT.NUM_PTS,
            kernel_size=FINAL_CONV_KERNEL, 
            stride=1,
            padding=1 if FINAL_CONV_KERNEL == 3 else 0
        )

        if 'epipolarpose' in cfg.BACKBONE.BODY:
            if cfg.EPIPOLAR.MERGE == 'both':
                self.epipolar_sampler1 = Epipolar()
            self.epipolar_sampler = Epipolar()
        else:
            self.epipolar_sampler = None
            self.epipolar_sampler1 = None

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=self.BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=self.BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, other_inputs=[None, None, None, None, None, None, None]):
        batch_size = x.shape[0]
        other_features, other_KRT, other_heatmaps, KRT, camera, other_camera, other_img = other_inputs
        features, heatmaps, batch_locs, batch_scos, corr_poss, depths = [], [], [], [], [], []
        # 3 x 256 x 256
        x = self.conv1(x)
        # 128 x 128
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        # 256 x 64 x 64

        def getOtherFeat(feat, sampler=None):
            # skip feature aggregation for last layer
            corr_pos = None
            depth = None
            if other_features is None:
                # normal hourglass
                return feat, None, None, None
            if 'epipolarpose' in cfg.BACKBONE.BODY:
                ret, corr_pos, depth, sample_locs = \
                    sampler(feat, other_features, KRT, other_KRT, \
                        camera=camera, other_camera=other_camera)
            return ret + feat, corr_pos, depth, sample_locs
            
        if cfg.EPIPOLAR.MERGE == 'early':
            feature = x
            x, corr_pos, depth, sample_locs = getOtherFeat(feature, sampler=self.epipolar_sampler)
            depths.append(depth)
            corr_poss.append(corr_pos)
        elif cfg.EPIPOLAR.MERGE == 'both':
            feature = x
            x, _, _, _ = getOtherFeat(feature, sampler=self.epipolar_sampler)

        x = self.layer2(x)
        #  512 x 32 × 32
        x = self.layer3(x)
        #   1024 x 16 × 16
        x = self.layer4(x)
        # 2048 x 8 x 8

        feature = self.deconv_layers(x)
        #256 x 64 x 64
        
        if cfg.EPIPOLAR.MERGE == 'late':
            x, corr_pos, depth, sample_locs = getOtherFeat(feature, sampler=self.epipolar_sampler)
            depths.append(depth)
            corr_poss.append(corr_pos)
        elif cfg.EPIPOLAR.MERGE == 'both':
            x, corr_pos, depth, sample_locs = getOtherFeat(feature, sampler=self.epipolar_sampler1)       
            depths.append(depth)
            corr_poss.append(corr_pos)                 
        else:
            x = feature

        #20 x 64 x 64
        heatmaps.append(self.final_layer(x))
        
        # The location of the current batch
        for ibatch in range(batch_size):
            batch_location, batch_score = find_tensor_peak_batch(heatmaps[-1][ibatch], 
                cfg.KEYPOINT.SIGMA, 
                cfg.BACKBONE.DOWNSAMPLE)
            batch_locs.append(batch_location)
            batch_scos.append(batch_score)
        batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)
        if other_features is None:
            corr_poss, depths = None, None
        else:
            corr_poss = corr_poss[-1]
            depths = depths[-1]

        return feature, heatmaps, batch_locs, batch_scos, corr_poss, depths, sample_locs, None 

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            if isinstance(pretrained, str) and os.path.isfile(pretrained):
                logger.info('=> loading pretrained model {}'.format(pretrained))
                pretrained_state_dict = torch.load(pretrained)
            else:
                logger.info('=> loading pretrained model from web')
                pretrained_state_dict = pretrained

            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)
            #load_state_dict(self, pretrained_state_dict, prefix='resnet.')
            #load_state_dict(self, pretrained_state_dict, prefix='backbone.')
            load_state_dict(self, pretrained_state_dict, strict=False, ignored_layers=['final_layer.bias', 'final_layer.weight'], prefix=cfg.WEIGHTS_PREFIX, prefix_replace=cfg.WEIGHTS_PREFIX_REPLACE)
            #self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)


resnet_spec = {'18': (BasicBlock, [2, 2, 2, 2]),
               '34': (BasicBlock, [3, 4, 6, 3]),
               '50': (Bottleneck, [3, 4, 6, 3]),
               '101': (Bottleneck, [3, 4, 23, 3]),
               '152': (Bottleneck, [3, 8, 36, 3])}

@registry.BACKBONES.register('poseR-18')
@registry.BACKBONES.register('poseR-34')
@registry.BACKBONES.register('poseR-50')
@registry.BACKBONES.register('poseR-101')
@registry.BACKBONES.register('poseR-152')
@registry.BACKBONES.register('epipolarposeR-18')
@registry.BACKBONES.register('epipolarposeR-34')
@registry.BACKBONES.register('epipolarposeR-50')
@registry.BACKBONES.register('epipolarposeR-101')
@registry.BACKBONES.register('epipolarposeR-152')
def get_pose_net(cfg, **kwargs):
    num_layers = cfg.BACKBONE.BODY.split('-')[-1]

    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, cfg, **kwargs)

    if cfg.BACKBONE.PRETRAINED:
        # model.init_weights(cfg.NETWORK.PRETRAINED)
        if cfg.BACKBONE.PRETRAINED_WEIGHTS:
            model.init_weights(cfg.BACKBONE.PRETRAINED_WEIGHTS)
        else:
            model.init_weights(model_zoo.load_url(model_urls['resnet'+num_layers]))

    return model
