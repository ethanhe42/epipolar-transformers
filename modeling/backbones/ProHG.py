# Stacked Hourglass Networks for Human Pose Estimation (https://arxiv.org/abs/1603.06937)
import time, math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_batch import find_tensor_peak_batch

from modeling import registry
from modeling.layers.meta import Meta
from modeling.layers.epipolar import Epipolar

from core import cfg

# import encoding.nn as enn

__all__ = ['hourglass', 'hourglass1', 'hourglass11']

class Residual(nn.Module):
    def __init__(self, numIn, numOut):
        super(Residual, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        middle = self.numOut // 2

        # self.bn = enn.SyncBatchNorm if cfg.BACKBONE.SYNC_BN else nn.BatchNorm2d
        self.bn = nn.BatchNorm2d

        self.conv_A = nn.Sequential(
                self.bn(numIn), nn.ReLU(inplace=True),
                nn.Conv2d(numIn, middle, kernel_size=1, dilation=1, padding=0, bias=True))
        self.conv_B = nn.Sequential(
                self.bn(middle), nn.ReLU(inplace=True),
                nn.Conv2d(middle, middle, kernel_size=3, dilation=1, padding=1, bias=True))
        self.conv_C = nn.Sequential(
                self.bn(middle), nn.ReLU(inplace=True),
                nn.Conv2d(middle, numOut, kernel_size=1, dilation=1, padding=0, bias=True))

        if self.numIn != self.numOut:
            self.branch = nn.Sequential(
                self.bn(self.numIn), nn.ReLU(inplace=True),
                nn.Conv2d(self.numIn, self.numOut, kernel_size=1, dilation=1, padding=0, bias=True))

    def forward(self, x):
        residual = x
        main = self.conv_A(x)
        main = self.conv_B(main)
        main = self.conv_C(main)
        if hasattr(self, 'branch'):
            residual = self.branch( residual )
        return main + residual


class HierarchicalPMS(nn.Module):
    def __init__(self, numIn, numOut):
        super(HierarchicalPMS, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        cA, cB, cC = self.numOut//2, self.numOut//4, self.numOut-self.numOut//2-self.numOut//4
        assert cA + cB + cC == numOut, '({:}, {:}, {:}) = {:}'.format(cA, cB, cC, numOut)

        # self.bn = enn.SyncBatchNorm if cfg.BACKBONE.SYNC_BN else nn.BatchNorm2d
        self.bn = nn.BatchNorm2d

        self.conv_A = nn.Sequential(
                self.bn(numIn), nn.ReLU(inplace=True),
                nn.Conv2d(numIn, cA, kernel_size=3, dilation=1, padding=1, bias=True))
        self.conv_B = nn.Sequential(
                self.bn(cA), nn.ReLU(inplace=True),
                nn.Conv2d(cA, cB, kernel_size=3, dilation=1, padding=1, bias=True))
        self.conv_C = nn.Sequential(
                self.bn(cB), nn.ReLU(inplace=True),
                nn.Conv2d(cB, cC, kernel_size=3, dilation=1, padding=1, bias=True))

        if self.numIn != self.numOut:
            self.branch = nn.Sequential(
                self.bn(self.numIn), nn.ReLU(inplace=True),
                nn.Conv2d(self.numIn, self.numOut, kernel_size=1, dilation=1, padding=0, bias=True))

    def forward(self, x):
        residual = x
        A = self.conv_A(x)
        B = self.conv_B(A)
        C = self.conv_C(B)
        main = torch.cat((A, B, C), dim=1)
        if hasattr(self, 'branch'):
            residual = self.branch( residual )
        return main + residual



class Hourglass(nn.Module):
    def __init__(self, n, nModules, nFeats, module):
        super(Hourglass, self).__init__()
        self.n = n
        self.nModules = nModules
        self.nFeats = nFeats
        
        self.res = nn.Sequential(*[module(nFeats, nFeats) for _ in range(nModules)])

        down = [nn.MaxPool2d(kernel_size = 2, stride = 2)]
        down += [module(nFeats, nFeats) for _ in range(nModules)]
        self.down = nn.Sequential(*down)

        if self.n > 1:
            self.mid = Hourglass(n - 1, self.nModules, self.nFeats, module)
        else:
            self.mid = nn.Sequential(*[module(nFeats, nFeats) for _ in range(nModules)])
        
        up = [module(nFeats, nFeats) for _ in range(nModules)]
        #up += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
        self.up = nn.Sequential(*up)
        
    def forward(self, x):
        res  = self.res(x)
        down = self.down(res)
        mid  = self.mid(down)
        up   = self.up(mid)
        up   = torch.nn.functional.interpolate(up, [res.size(2), res.size(3)], mode='bilinear', align_corners=True)
        return res + up



class HourGlassNet(nn.Module):
    def __init__(self, config, points, sigma, input_dim):
        super(HourGlassNet, self).__init__()
        self.sigma      = sigma

        if config['module'] == 'Residual':
            module = Residual
        elif config['module'] == 'HierarchicalPMS':
            module = HierarchicalPMS
        else:
            raise ValueError('Invaliad module for HG : {:}'.format(config['module']))

        self.pts_num  = points
        self.downsample = config['downsample']
        self.nStack   = config['stages']
        self.nModules = config['nModules']
        self.nFeats   = config['nFeats']
        self.recursive = config['recursive']
        # self.bn = enn.SyncBatchNorm if cfg.BACKBONE.SYNC_BN else nn.BatchNorm2d
        self.bn = nn.BatchNorm2d

        #self.conv = nn.Sequential(
        #              nn.Conv2d(input_dim, 64, kernel_size = 7, stride = 2, padding = 3, bias = True), 
        #              nn.BatchNorm2d(64), nn.ReLU(inplace = True))
        self.conv = nn.Sequential(
                nn.Conv2d(input_dim, 32, kernel_size = 3, stride = 2, padding = 1, bias = True), 
                self.bn(32), nn.ReLU(inplace = True),
                nn.Conv2d(       32, 32, kernel_size = 3, stride = 1, padding = 1, bias = True), 
                self.bn(32), nn.ReLU(inplace = True),
                nn.Conv2d(       32, 64, kernel_size = 3, stride = 1, padding = 1, bias = True), 
                self.bn(64), nn.ReLU(inplace = True))
        
        self.ress = nn.Sequential(
                module(64, 128),
                nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
                module(128, 128), module(128, self.nFeats))
        
        _features, _tmpOut, _ll_, _tmpOut_ = [], [], [], []

        for i in range(self.nStack):
            feature = Hourglass(self.recursive, self.nModules, self.nFeats, module)
            feature = [feature] + [module(self.nFeats, self.nFeats) for _ in range(self.nModules)]
            feature += [nn.Conv2d(self.nFeats, self.nFeats, kernel_size = 1, stride = 1, bias = True),
                                    self.bn(self.nFeats), nn.ReLU(inplace = True)]
            feature = nn.Sequential(*feature)
            _features.append(feature)
            _tmpOut.append(nn.Conv2d(self.nFeats, self.pts_num, kernel_size = 1, stride = 1, bias = True))
            if i < self.nStack - 1:
                _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, kernel_size = 1, stride = 1, bias = True))
                _tmpOut_.append(nn.Conv2d(self.pts_num, self.nFeats, kernel_size = 1, stride = 1, bias = True))
                
        self.features = nn.ModuleList(_features)
        self.tmpOuts = nn.ModuleList(_tmpOut)
        self.trsfeas = nn.ModuleList(_ll_)
        self.trstmps = nn.ModuleList(_tmpOut_)
        if config['sigmoid']:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None
    
        if 'metaHG' in cfg.BACKBONE.BODY:
            _meta = []
            for i in range(self.nStack):
                _meta.append(Meta(self.nFeats))
            self.meta = nn.ModuleList(_meta)
            self.testmeta = Meta(self.nFeats)
        elif 'epipolarHG' in cfg.BACKBONE.BODY:
            self.epipolar_sampler = Epipolar()

        self.avgpool = nn.AvgPool2d(kernel_size=4)

    def extra_repr(self):
        return ('{name}(sigma={sigma}, downsample={downsample})'.format(name=self.__class__.__name__, **self.__dict__))

    
    def forward(self, inputs, other_inputs=[None, None, None, None, None, None, None]):
        assert inputs.dim() == 4, 'This model accepts 4 dimension input tensor: {}'.format(inputs.size())
        batch_size, feature_dim = inputs.shape[0], inputs.shape[1]
        other_features, other_KRT, other_heatmaps, KRT, camera, other_camera, other_img = other_inputs
        x = self.conv(inputs)
        x = self.ress(x)
        
        features, heatmaps, batch_locs, batch_scos, corr_poss, depths = [], [], [], [], [], []
        
        def getOtherFeat(i, feat):
            # skip feature aggregation for last layer
            corr_pos = None
            depth = None
            sample_locs = None
            if other_features is None:
                # normal hourglass
                return feat, None, None, None
            if i + 1 == self.nStack and cfg.EPIPOLAR.MERGE == 'late':
                ret = 0
            if 'simplemultiviewHG' in cfg.BACKBONE.BODY:
                ret = other_features[i]
            if 'metaHG' in cfg.BACKBONE.BODY:
                ret = self.meta[i](KRT, other_KRT, other_features[i])
            if 'epipolarHG' in cfg.BACKBONE.BODY:
                if cfg.EPIPOLAR.FIND_CORR == 'feature':
                    ret, corr_pos, depth, sample_locs = \
                        self.epipolar_sampler(feat, other_features[i], KRT, other_KRT, camera=camera, other_camera=other_camera, ref1=feat, ref2=other_features[i])
                elif cfg.EPIPOLAR.FIND_CORR == 'rgb':
                    assert not cfg.EPIPOLAR.PRIOR
                    downsampled_img1 = self.avgpool(inputs)
                    downsampled_img2 = self.avgpool(other_img)
                    downsampled_img1.detach()
                    downsampled_img2.detach()

                    ret, corr_pos, depth, sample_locs = \
                        self.epipolar_sampler(feat, other_features[i], KRT, other_KRT, camera=camera, other_camera=other_camera, ref1=downsampled_img1, ref2=downsampled_img2)
        
                # print(ret)
                # print(feat.abs().mean(), ret.abs().mean())
            if cfg.EPIPOLAR.OTHER_ONLY:
                return ret, corr_pos, depth, sample_locs
            return ret + feat, corr_pos, depth, sample_locs

        feat_cnt = 0
        for i in range(self.nStack):
            if cfg.EPIPOLAR.MERGE == 'early':
                other_feature_T, corr_pos, depth, sample_locs = getOtherFeat(feat_cnt, x)
                feat_cnt += 1
                if cfg.SOLVER.FINETUNE:
                    other_feature_T.detach_()

                feature = self.features[i](other_feature_T)
                features.append(x)
                depths.append(depth)
                corr_poss.append(corr_pos)
            elif cfg.EPIPOLAR.MERGE == 'late':
                feature = self.features[i](x)
                if cfg.SOLVER.FINETUNE:
                    feature.detach_()                
                feature, corr_pos, depth, sample_locs = getOtherFeat(feat_cnt, feature)
                feat_cnt += 1
                features.append(feature)
                depths.append(depth)
                corr_poss.append(corr_pos)
            elif cfg.EPIPOLAR.MERGE == 'both':
                other_feature_T, corr_pos, depth, sample_locs = getOtherFeat(feat_cnt, x)
                depths.append(depth)
                corr_poss.append(corr_pos)
                features.append(x)
                feat_cnt += 1
                if cfg.SOLVER.FINETUNE:
                    other_feature_T.detach_()
                feature = self.features[i](other_feature_T)
                feature, corr_pos, depth, sample_locs = getOtherFeat(feat_cnt, feature)
                feat_cnt += 1
                features.append(feature)
                depths.append(depth)
                corr_poss.append(corr_pos)
            elif cfg.EPIPOLAR.MERGE == 'none':
                features.append(x)
            else:
                raise NotImplementedError

            tmpOut = self.tmpOuts[i](feature)
            if self.sigmoid is not None:
                tmpOut = self.sigmoid(tmpOut)
            heatmaps.append(tmpOut)
            if i < self.nStack - 1:
                ll_ = self.trsfeas[i](feature)
                tmpOut_ = self.trstmps[i](tmpOut)
                x = x + ll_ + tmpOut_

        if cfg.EPIPOLAR.WARPEDHEATMAP and other_heatmaps is not None:
            warpedheatmap, _, _, _ = self.epipolar_sampler(None, other_heatmaps, KRT, other_KRT, depths[0])
        else:
            warpedheatmap = None

        # The location of the current batch
        for ibatch in range(batch_size):
            batch_location, batch_score = find_tensor_peak_batch(heatmaps[-1][ibatch], self.sigma, self.downsample)
            batch_locs.append(batch_location)
            batch_scos.append(batch_score)
        batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)
        if other_features is None:
            corr_poss, depths = None, None
        else:
            corr_poss = corr_poss[-1]
            depths = depths[-1]

        return features, heatmaps, batch_locs, batch_scos, corr_poss, depths, sample_locs, warpedheatmap


@registry.BACKBONES.register('HG')
@registry.BACKBONES.register('simplemultiviewHG')
@registry.BACKBONES.register('metaHG')
@registry.BACKBONES.register('epipolarHG')
@registry.BACKBONES.register('metaepipolarHG')
def hourglass(cfg, **kwargs):
        """Constructs a Hourglass model.
        Args:
                pretrained (bool): If True, returns a model pre-trained
        """
        config = {
            "stages"    : 3,
            "nModules"  : 1,
            "recursive" : 3,
            "nFeats"    : cfg.KEYPOINT.NFEATS,
            "module"    : "Residual",
            "background": 1,
            "sigmoid"   : 0,
            "downsample": 4,
        }
        idim = 3
        sigma = cfg.KEYPOINT.SIGMA
        points = cfg.KEYPOINT.NUM_PTS
        model = HourGlassNet(config, points, sigma, idim)
        if cfg.BACKBONE.PRETRAINED:
                raise NotImplementedError
                #model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
        return model

@registry.BACKBONES.register('HG1')
@registry.BACKBONES.register('simplemultiviewHG1')
@registry.BACKBONES.register('metaHG1')
@registry.BACKBONES.register('epipolarHG1')
@registry.BACKBONES.register('metaepipolarHG1')
def hourglass1(cfg, **kwargs):
        """Constructs a Hourglass model with 1 stage.
        Args:
                pretrained (bool): If True, returns a model pre-trained
        """
        config = {
            "stages"    : 1,
            "nModules"  : 1,
            "recursive" : 3,
            "nFeats"    : cfg.KEYPOINT.NFEATS,
            "module"    : "Residual",
            "background": 1,
            "sigmoid"   : 0,
            "downsample": 4,
        }
        idim = 3
        sigma = cfg.KEYPOINT.SIGMA
        points = cfg.KEYPOINT.NUM_PTS
        model = HourGlassNet(config, points, sigma, idim)
        if cfg.BACKBONE.PRETRAINED:
                raise NotImplementedError
                #model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
        return model

@registry.BACKBONES.register('HG11')
@registry.BACKBONES.register('simplemultiviewHG11')
@registry.BACKBONES.register('metaHG11')
@registry.BACKBONES.register('epipolarHG11')
@registry.BACKBONES.register('metaepipolarHG11')
def hourglass11(cfg, **kwargs):
        """Constructs a Hourglass model with 1 stage.
        Args:
                pretrained (bool): If True, returns a model pre-trained
        """
        config = {
            "stages"    : 1,
            "nModules"  : 1,
            "recursive" : 1,
            "nFeats"    : cfg.KEYPOINT.NFEATS,
            "module"    : "Residual",
            "background": 1,
            "sigmoid"   : 0,
            "downsample": 4,
        }
        idim = 3
        sigma = cfg.KEYPOINT.SIGMA
        points = cfg.KEYPOINT.NUM_PTS
        model = HourGlassNet(config, points, sigma, idim)
        if cfg.BACKBONE.PRETRAINED:
                raise NotImplementedError
                #model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
        return model
