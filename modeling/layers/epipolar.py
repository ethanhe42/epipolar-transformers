import torch
from torch import nn
import torch.nn.functional as F

from core import cfg
from vision.multiview import camera_center, normalize, de_normalize, pix2coord, coord2pix, findFundamentalMat
from .BN import zeroinitBN

visualize_prob = 0.1

class Epipolar(nn.Module):
    def __init__(self, debug=False):
        super(Epipolar, self).__init__()
        self.debug = debug
        self.downsample = cfg.BACKBONE.DOWNSAMPLE
        # h, w = cfg.DATASETS.IMAGE_SIZE 
        # self.feat_h, self.feat_w = h // self.downsample, w // self.downsample
        self.feat_h, self.feat_w = cfg.KEYPOINT.HEATMAP_SIZE
        self.sample_size = cfg.EPIPOLAR.SAMPLESIZE
        self.epsilon = 0.001 # for avoiding floating point error

        y = torch.arange(0, self.feat_h, dtype=torch.float) # 0 .. 128
        x = torch.arange(0, self.feat_w, dtype=torch.float) # 0 .. 84

        if cfg.EPIPOLAR.REPROJECT_LOSS_WEIGHT > 0:
            gt_grid_y, gt_grid_x = torch.meshgrid(y, x)
            self.gt_grid = torch.stack((gt_grid_x, gt_grid_y), -1)[None, ...]
            self.gt_grid = normalize(self.gt_grid, self.feat_h, self.feat_w)
        #THESE ARE WRONG
        # y = y * self.downsample + self.downsample / 2.0 - 0.5 # * 4 + 2 -0.5
        # x = x * self.downsample + self.downsample / 2.0 - 0.5
        # # scale back to original image
        # y = y * cfg.DATASETS.IMAGE_RESIZE * cfg.DATASETS.PREDICT_RESIZE # 2 * 4
        # x = x * cfg.DATASETS.IMAGE_RESIZE * cfg.DATASETS.PREDICT_RESIZE
        y = pix2coord(y, cfg.BACKBONE.DOWNSAMPLE)
        y = y * cfg.DATASETS.IMAGE_RESIZE * cfg.DATASETS.PREDICT_RESIZE
        x = pix2coord(x, cfg.BACKBONE.DOWNSAMPLE)   # 128 -> 512
        x = x * cfg.DATASETS.IMAGE_RESIZE * cfg.DATASETS.PREDICT_RESIZE # ->1024->4096

        grid_y, grid_x = torch.meshgrid(y, x)
        # grid_y: 84x128
        # 3 x HW
        # TODO check whether yx or xy
        self.grid = torch.stack((grid_x, grid_y, torch.ones_like(grid_x))).view(3, -1)

        self.xmin = x[0]
        self.ymin = y[0]
        self.xmax = x[-1]
        self.ymax = y[-1]
        self.tmp_tensor = torch.tensor([True, True, False, False])
        self.outrange_tensor = torch.tensor([
            self.xmin-10000, self.ymin-10000, 
            self.xmin-10000, self.ymin-10000]).view(2, 2)
        self.sample_steps = torch.range(0, 1, 1./(self.sample_size-1)).view(-1, 1, 1, 1)

        if cfg.EPIPOLAR.BOTTLENECK != 1:
            assert 'z' in cfg.EPIPOLAR.PARAMETERIZED
            assert 'theta' in cfg.EPIPOLAR.PARAMETERIZED
            assert 'phi' in cfg.EPIPOLAR.PARAMETERIZED
            assert 'g' in cfg.EPIPOLAR.PARAMETERIZED
            assert not cfg.EPIPOLAR.ZRESIDUAL

        if 'z' in cfg.EPIPOLAR.PARAMETERIZED:
            self.z = nn.Conv2d(cfg.KEYPOINT.NFEATS // cfg.EPIPOLAR.BOTTLENECK, cfg.KEYPOINT.NFEATS, kernel_size=1, stride=1, padding=0, bias=True)
            self.bn = zeroinitBN(cfg.KEYPOINT.NFEATS)
        if 'theta' in cfg.EPIPOLAR.PARAMETERIZED:
            self.theta = nn.Conv2d(cfg.KEYPOINT.NFEATS, cfg.KEYPOINT.NFEATS // cfg.EPIPOLAR.BOTTLENECK, kernel_size=1, stride=1, padding=0, bias=True)
        if 'phi' in cfg.EPIPOLAR.PARAMETERIZED:
            self.phi = nn.Conv2d(cfg.KEYPOINT.NFEATS, cfg.KEYPOINT.NFEATS // cfg.EPIPOLAR.BOTTLENECK, kernel_size=1, stride=1, padding=0, bias=True)
        if 'g' in cfg.EPIPOLAR.PARAMETERIZED:
            self.g = nn.Conv2d(cfg.KEYPOINT.NFEATS, cfg.KEYPOINT.NFEATS // cfg.EPIPOLAR.BOTTLENECK, kernel_size=1, stride=1, padding=0, bias=True)
        
        if cfg.EPIPOLAR.PRIOR:
            self.prior = {}
            for i in cfg.DATASETS.CAMERAS:
                for j in cfg.DATASETS.CAMERAS:
                    if j == i:
                        continue
                    self.prior[(i, j)] = nn.Parameter(torch.Tensor(self.sample_size, self.feat_h, self.feat_w))
                    self.prior[(i, j)].data.uniform_(0, 0.1)

    def forward(self, feat1, feat2, P1, P2, depth=None, camera=None, other_camera=None, ref1=None, ref2=None):
        """ 
        Args:
            feat1         : N x C x H x W
            feat2         : N x C x H x W
            P1          : N x 3 x 4
            P2          : N x 3 x 4
        1. Compute epipolar lines: NHW x 3 (http://users.umiacs.umd.edu/~ramani/cmsc828d/lecture27.pdf)
        2. Compute intersections with the image: NHW x 2 x 2
            4 intersections with each boundary of the image NHW x 4 x 2
            Convert to (-1, 1)
            find intersections on the rectangle NHW x 4 T/F, NHW x 2 x 2
            sample N*sample_size x H x W x 2
                if there's no intersection, the sample points are out of (-1, 1), therefore ignored by pytorch
        3. Sample points between the intersections: sample_size x N x H x W x 2
        4. grid_sample: sample_size*N x C x H x W -> sample_size x N x C x H x W
            trick: compute feat1 feat2 dot product first: N x HW x H x W
        5. max pooling/attention: N x C x H x W
        """
        if depth is None:
            usedepth = False
        else:
            usedepth = True
        # if cfg.EPIPOLAR.PARAMETERIZED and not usedepth:

        assert cfg.EPIPOLAR.ATTENTION in {'avg', 'max'}
        assert cfg.EPIPOLAR.SIMILARITY in {'cos', 'dot', 'prior'}
        # if cfg.EPIPOLAR.ATTENTION == 'avg':
        #     assert 'phi' not in cfg.EPIPOLAR.PARAMETERIZED
        #     assert 'theta' not in cfg.EPIPOLAR.PARAMETERIZED
        #     assert 'g' not in cfg.EPIPOLAR.PARAMETERIZED

        #     # if 'g' in cfg.EPIPOLAR.PARAMETERIZED:
        #     #     gout = self.g(feat2)
        #     # else:
        #     #     gout = feat2
            
        #     if 'other2' in cfg.EPIPOLAR.OTHER_GRAD:
        #         gout = gout
        #     else:
        #         gout = gout.detach()

        #     # if 'phi' in cfg.EPIPOLAR.PARAMETERIZED:
        #     #     feat2 = self.phi(feat2)
        #     # if 'theta' in cfg.EPIPOLAR.PARAMETERIZED:
        #     #     feat1 = self.theta(feat1)
        # else:

        # assert 'phi' not in cfg.EPIPOLAR.PARAMETERIZED
        # assert 'theta' not in cfg.EPIPOLAR.PARAMETERIZED
        # assert 'g' not in cfg.EPIPOLAR.PARAMETERIZED

        if cfg.EPIPOLAR.FIND_CORR == 'rgb':
            assert ref1 is not None and ref2 is not None
            other1 = ref2
            other1.detach()
            assert 'other1' not in cfg.EPIPOLAR.OTHER_GRAD
            assert 'phi' not in cfg.EPIPOLAR.PARAMETERIZED
        else:
            if 'other1' in cfg.EPIPOLAR.OTHER_GRAD:
                other1 = feat2
            else:
                other1 = feat2.detach()
            if 'phi' in cfg.EPIPOLAR.PARAMETERIZED:
                other1 = self.phi(other1)
            if 'theta' in cfg.EPIPOLAR.PARAMETERIZED:
                feat1 = self.theta(feat1)

        if 'other2' in cfg.EPIPOLAR.OTHER_GRAD:
            other2 = feat2
        else:
            other2 = feat2.detach()

        if 'g' in cfg.EPIPOLAR.PARAMETERIZED:
            other2 = self.g(other2)
        

        if feat1 is None:
            N, C, H, W = feat2.shape
        else:
            N, C, H, W = feat1.shape
        if ref2 is None:
            ref2 = other1

        C_ref = ref2.shape[1]
        other2 = other2.view(1, N, C, H, W)
        other2 = other2.expand(self.sample_size, N, C, H, W)

        other1 = other1.view(1, N, C_ref, H, W) # ref2
        other1 = other1.expand(self.sample_size, N, C_ref, H, W)                      

       # if ref1 is not None:
       #     ref1 = ref1.view(1, N, C_ref, H, W)
       #     ref1 = ref1.expand(self.sample_size, N, C_ref, H, W)                     
       #     print(ref1.shape, feat1.shape)
            
        with torch.no_grad():
            if self.debug:
                sample_locs, intersections, mask, valid_intersections, start, vec = self.grid2sample_locs(self.grid, P1, P2, H, W)
            else:
                sample_locs = self.grid2sample_locs(self.grid, P1, P2, H, W)
            sample_locs = sample_locs.float()
        out = []
        corr_pos = []
        if not usedepth:
            depth = []
        for i in range(N):
            # sample_size x C x H x W
            # feat2: sample_size x N x C x H x W
            # feat2[:, i]: sample_size x C x H x W
            # sample_locs[:, i]: sample_size x H x W x 2

            # ---
            # ref2: sample_size x N x 3 x H x W
            # ref1: sample_size x N x 3 x H x W

            # tmp sample_size x C x H x W
            other1_sampled = F.grid_sample(other1[:, i], sample_locs[:, i])
            if cfg.EPIPOLAR.POOLING:
                stride = 2
                other1_sampled, _ = other1_sampled.view(stride, self.sample_size // stride, C, H, W).max(0)

            if other1 is other2:
                assert not cfg.EPIPOLAR.FIND_CORR == 'rgb'
                assert 'other2' in cfg.EPIPOLAR.OTHER_GRAD and 'other1' in cfg.EPIPOLAR.OTHER_GRAD, "other1 is other 2??"
                # other1 and other2 is the same thing
                other2_sampled = other1_sampled
            else:
                other2_sampled = F.grid_sample(other2[:, i], sample_locs[:, i])
                if cfg.EPIPOLAR.POOLING:
                    stride = 2
                    other2_sampled, _ = other2_sampled.view(stride, self.sample_size // stride, C, H, W).max(0)

            if usedepth:
                sim = depth[i]
            else:
                if cfg.EPIPOLAR.PRIOR:
                    sim = self.epipolar_similarity(feat1[i], other1_sampled, 
                        int(camera[i].item()), int(other_camera[i].item()))
                elif cfg.EPIPOLAR.FIND_CORR == 'rgb':
                    sim = self.epipolar_similarity(ref1[i], other1_sampled)
                else:
                    sim = self.epipolar_similarity(feat1[i], other1_sampled)
            if cfg.EPIPOLAR.ATTENTION == 'max':
                # 1 C H W
                idx = sim.argmax(0)
                with torch.no_grad():
                    # H x W x 2
                    pos = torch.gather(sample_locs[:, i], 0, idx.view(1, H, W, 1).expand(-1, -1, -1, 2)).squeeze()
                    pos = de_normalize(pos, H, W)
                    corr_pos.append(pos)
                idx = idx.view(1, 1, H, W).expand(-1, C, -1, -1)
                # C x H x W
                tmp = torch.gather(other2_sampled, 0, idx).squeeze()
            elif cfg.EPIPOLAR.ATTENTION == 'avg':
                idx = sim.argmax(0)
                with torch.no_grad():
                    # H x W x 2
                    pos = torch.gather(sample_locs[:, i], 0, idx.view(1, H, W, 1).expand(-1, -1, -1, 2)).squeeze()
                    pos = de_normalize(pos, H, W)
                    corr_pos.append(pos)
                tmp = (other2_sampled * sim.view(-1, 1, H, W)).sum(0)
            if not usedepth:
                depth.append(sim) 

            out.append(tmp)
        out = torch.stack(out)
        if 'z' in cfg.EPIPOLAR.PARAMETERIZED and not usedepth:
            finalout = self.z(out)
            finalout = self.bn(finalout)
            if cfg.EPIPOLAR.ZRESIDUAL:
                finalout = finalout + out
        else:
            finalout = out

        if feat1 is not None and cfg.EPIPOLAR.REPROJECT_LOSS_WEIGHT != 0:
            reprojected_locs = self.reproject(feat1, feat2, depth, sample_locs, P1, P2)
            with torch.no_grad():
                reproject_mask = ((reprojected_locs.min(-1)[0] > -1) & (reprojected_locs.max(-1)[0] < 1)).view(N, H, W, 1)
            return finalout, depth, reprojected_locs, self.gt_grid.expand(N, -1, -1, -1).to(reprojected_locs), reproject_mask
        corr_pos = torch.stack(corr_pos)
        depth = torch.stack(depth)
        if self.debug:
            return finalout, corr_pos, depth, sample_locs, intersections, mask, valid_intersections, start, vec
        if cfg.VIS.EPIPOLAR_LINE:
            return finalout, corr_pos, depth, sample_locs.transpose(0,1)
        else:
            return finalout, corr_pos, depth, None

    
    def epipolar_similarity(self, feat1, sampled_feat2, cam1=None, cam2=None):
        """ 
        Args:
            fea1: C, H, W
            sampled_feat2: sample_size, C, H, W
        Return:
            sim: sample_size H W
        """
        C, H, W = feat1.shape
        sample_size = sampled_feat2.shape[0]
        if cfg.EPIPOLAR.ATTENTION == 'max':
            # sample_size H W
            sim = F.cosine_similarity(
                feat1.view(1, C, H, W).expand(sample_size, -1, -1, -1),
                sampled_feat2, 1)
        elif cfg.EPIPOLAR.ATTENTION == 'avg':
            if cfg.EPIPOLAR.SIMILARITY == 'prior':
                return self.prior[(cam1, cam2)].to(feat1)
            elif cfg.EPIPOLAR.SIMILARITY == 'cos':
                sim = F.cosine_similarity(
                    feat1.view(1, C, H, W).expand(sample_size, -1, -1, -1),
                    sampled_feat2, 1)
            elif cfg.EPIPOLAR.SIMILARITY == 'dot':
                sim = (sampled_feat2 * feat1.view(1, C, H, W).expand(sample_size, -1, -1, -1)).sum(1)
            else:
                raise NotImplementedError
            sim[sim==0] = -1e10

            if cfg.EPIPOLAR.PRIOR and not cfg.EPIPOLAR.PRIORMUL:
                sim = sim + self.prior[(cam1, cam2)].to(sim)

            if cfg.EPIPOLAR.SOFTMAX_ENABLED:
                # following https://arxiv.org/pdf/1706.03762.pdf d_k
                # TODO: the result is bad
                sim = sim * cfg.EPIPOLAR.SOFTMAXSCALE 
                sim = F.softmax(sim, 0)
                if cfg.EPIPOLAR.PRIORMUL:
                    sim = sim * self.prior[(cam1, cam2)].to(sim)
            else:
                sim /= sample_size
            # if 0:
            #     rmax, rmin, cnt =0, 0, 0
            #     for i in range(sim.shape[1]):
            #         for j in range(sim.shape[2]):
            #             # print(sim[:, i, j])
            #             rmax += sim[:, i, j].max().item()
            #             rmin += sim[:, i, j].min().item()
            #             cnt += 1
            #     print('all', sim.max(), sim.min(), rmax/cnt, rmin/cnt)
        return sim

    def grid2sample_locs(self, grid, P1, P2, H, W):
        """ Get samples locs on the other view, given grid locs on ref view
        Args:
            grid: 3 x HW, real world xy (4096)
        Return:
            sample_locs: sample_size x N x H x W x 2, float xy (-1, 1)
        """
        N = P1.shape[0]

        # F = findFundamentalMat(P1, P2)
        # N x 4 x 3
        P1t = P1.transpose(1, 2)
        # P1inv = torch.matmul(P1t, torch.inverse(torch.matmul(P1, P1t)))
        P1inv = torch.stack([i.pinverse() for i in P1])
        # N x 4 x HW
        X = torch.matmul(P1inv, grid.to(P1inv))
        # N x 3 x HW
        x2 = torch.matmul(P2, X)
        #numerical stability        
        x2 /= x2[:, [2], :]
        # N x 4
        center, _ = camera_center(P1, engine='torch')
        # N x 3 x 1
        e2 = torch.matmul(P2, center).view(N, 3, 1)
        #numerical stability
        e2 /= e2[:, [2], :]
        # N x 3 x HW
        l2 = torch.cross(e2.expand_as(x2), x2, dim=1)
        # N x HW x 3
        l2 = l2.transpose(1, 2)

        xmin = self.xmin.to(l2)
        xmax = self.xmax.to(l2)
        ymin = self.ymin.to(l2)
        ymax = self.ymax.to(l2)
        # N x HW
        # ( 3 ]
        #⎴     ⏜
        #1     2
        #⏝     ⎵
        # [ 0 )
        # by1 = -(xmin * l2[..., 0] + l2[..., 2]) / l2[..., 1]
        # by2 = -(xmax * l2[..., 0] + l2[..., 2]) / l2[..., 1]
        # bx0 = -(ymin * l2[..., 1] + l2[..., 2]) / l2[..., 0]
        # bx3 = -(ymax * l2[..., 1] + l2[..., 2]) / l2[..., 0]
        #numerical stability
        EPS = torch.tensor(self.epsilon).to(l2)
        by1 = -(xmin * l2[..., 0] + l2[..., 2]) / (torch.sign(l2[..., 1]) * torch.max(torch.abs(l2[..., 1]), EPS))
        by2 = -(xmax * l2[..., 0] + l2[..., 2]) / (torch.sign(l2[..., 1]) * torch.max(torch.abs(l2[..., 1]), EPS))
        bx0 = -(ymin * l2[..., 1] + l2[..., 2]) / (torch.sign(l2[..., 0]) * torch.max(torch.abs(l2[..., 0]), EPS))
        bx3 = -(ymax * l2[..., 1] + l2[..., 2]) / (torch.sign(l2[..., 0]) * torch.max(torch.abs(l2[..., 0]), EPS))
        # N x HW x 4
        intersections = torch.stack((
            bx0,
            by1,
            by2,
            bx3,
            ), -1)
        # N x HW x 4 x 2
        intersections = intersections.view(N, H*W, 4, 1).repeat(1, 1, 1, 2)
        intersections[..., 0, 1] = ymin
        intersections[..., 1, 0] = xmin
        intersections[..., 2, 0] = xmax
        intersections[..., 3, 1] = ymax
        # N x HW x 4
        mask = torch.stack((
            (bx0 >= xmin + self.epsilon) & (bx0 <  xmax - self.epsilon),
            (by1 >  ymin + self.epsilon) & (by1 <= ymax - self.epsilon),
            (by2 >= ymin + self.epsilon) & (by2 <  ymax - self.epsilon),
            (bx3 >  xmin + self.epsilon) & (bx3 <= xmax - self.epsilon),
            ), -1)
        # N x HW
        Nintersections = mask.sum(-1)
        # rule out all lines have no intersections
        mask[Nintersections < 2] = 0
        tmp_mask = mask.clone()
        tmp_mask[Nintersections < 2] = self.tmp_tensor.to(tmp_mask)
        # assert (Nintersections <= 2).all().item(), intersections[Nintersections > 2]
        # N x HW x 2 x 2
        valid_intersections = intersections[tmp_mask].view(N, H*W, 2, 2)
        valid_intersections[Nintersections < 2] = self.outrange_tensor.to(valid_intersections)
        # N x HW x 2
        start = valid_intersections[..., 0, :]
        vec = valid_intersections[..., 1, :] - start
        vec = vec.view(1, N, H*W, 2)
        # sample_size x N x HW x 2
        sample_locs = start.view(1, N, H*W, 2) + vec * self.sample_steps.to(vec)
        # normalize
        sample_locs = sample_locs / cfg.DATASETS.IMAGE_RESIZE / cfg.DATASETS.PREDICT_RESIZE
        sample_locs = coord2pix(sample_locs, cfg.BACKBONE.DOWNSAMPLE)
        # sample_size*N x H x W x 2
        sample_locs = normalize(sample_locs, H, W).view(-1, H, W, 2)
        sample_locs = sample_locs.view(self.sample_size, N, H, W, 2)
        if self.debug:
            return sample_locs, intersections, mask, valid_intersections, start, vec
        return sample_locs

    def reproject(self, feat1, feat2, depth, sample_locs, P1s, P2s):
        """ 
        Args:
            feat1: N C H W
            feat2: N C H W
            depth: list of N [sample_size H W]
            sample_locs: sample_size x N x H x W x 2, (-1, 1)
            P1s: N x 3 x 4
            P2s: N x 3 x 4
        Return:
            out: N H W 2, (-1, 1)
        """
        N, C, H, W = feat1.shape
        # N H W 2 (pix)       sample_size N H W 2,  sample_size N H W
        expected_locs = torch.mul(sample_locs, torch.stack(depth, 1)[..., None]).sum(0)
        # N C H W
        matched_feat2 = F.grid_sample(feat2, expected_locs)
        # # N H W 3,   (-1, 1) -> 128
        expected_locs = de_normalize(expected_locs, H, W, engine='torch')
        expected_locs = pix2coord(expected_locs, cfg.BACKBONE.DOWNSAMPLE)   # 128 -> 512
        expected_locs = expected_locs * cfg.DATASETS.IMAGE_RESIZE * cfg.DATASETS.PREDICT_RESIZE # ->1024->4096
        expected_locs = torch.cat((expected_locs, 
            torch.ones((expected_locs.shape[0], expected_locs.shape[1], expected_locs.shape[2], 1)).to(expected_locs)
            ), 3)
        # N HW 3        
        expected_locs = expected_locs.view(expected_locs.shape[0], -1, 3)
        #N x sample_size x H x W x 2
        reproject_sample_locs = []
        for new_loc, P1, P2 in zip(expected_locs, P1s, P2s):
            #sample_size x H x W x 2 (-1,1)           2 HW
            reproject_sample_locs.append(self.grid2sample_locs(new_loc.t(), P2[None, ...], P1[None, ...], H, W).squeeze())
        # sample_size, N, C, H, W
        expanded_feat1 = feat1.view(1, N, C, H, W).expand(self.sample_size, N, C, H, W)
        out = []
        for i, reproject_sample_loc in enumerate(reproject_sample_locs):
            # sample_size, C, H, W   sample_size, C, H, W  sample_size x H x W x 2  
            tmp = F.grid_sample(expanded_feat1[:, i], reproject_sample_loc)
            # sample_size, H, W
            sim = self.epipolar_similarity(matched_feat2[i], tmp)
            # H W 2 = sample_size H W 2 * sample_size H W
            expected_reproject_loc = (reproject_sample_loc * sim.view(self.sample_size, H, W, 1)).sum(0)
            out.append(expected_reproject_loc)
        # N H W 2
        out = torch.stack(out)
        return out

        # loss = (out - self.gt_grid.to(out))
        # F.mse_loss(inputs, targets, reduction=self.reduction)


    def imgforward_withdepth(self, feat1, feat2, P1, P2, depth):
        """ 
        Args:
            feat1         : N x C x H x W
            feat2         : N x C x H x W
            P1          : N x 3 x 4
            P2          : N x 3 x 4
        1. Compute epipolar lines: NHW x 3 (http://users.umiacs.umd.edu/~ramani/cmsc828d/lecture27.pdf)
        2. Compute intersections with the image: NHW x 2 x 2
            4 intersections with each boundary of the image NHW x 4 x 2
            Convert to (-1, 1)
            find intersections on the rectangle NHW x 4 T/F, NHW x 2 x 2
            sample N*sample_size x H x W x 2
                if there's no intersection, the sample points are out of (-1, 1), therefore ignored by pytorch
        3. Sample points between the intersections: sample_size x N x H x W x 2
        4. grid_sample: sample_size*N x C x H x W -> sample_size x N x C x H x W
            trick: compute feat1 feat2 dot product first: N x HW x H x W
        5. max pooling/attention: N x C x H x W
        """
        if feat1 is None:
            N, C, H, W = feat2.shape
        else:
            N, C, H, W = feat1.shape        
        with torch.no_grad():
            sample_locs = self.grid2sample_locs(self.grid, P1, P2, H, W)
            feat2 = feat2.view(1, N, C, H, W)
            feat2 = feat2.expand(self.sample_size, N, C, H, W)
        out = []
        # m = nn.UpsamplingNearest2d(scale_factor=cfg.BACKBONE.DOWNSAMPLE)
        for i in range(N):
            # sample_size x C x H x W
            tmp = F.grid_sample(feat2[:, i], sample_locs[:, i])
            tmp[0] = 0.
            # if cfg.EPIPOLAR.ATTENTION == 'max':
            # sample_size H W
            # 1 C H W
            idx = depth.argmax(0)
            idx = idx.view(1, 1, H, W).expand(-1, C, -1, -1)
            # C x H x W
            tmp = torch.gather(tmp, 0, idx).squeeze()
            out.append(tmp)
        out = torch.stack(out)
        return out, sample_locs
        depth = torch.stack(depth)
        return out, depth
