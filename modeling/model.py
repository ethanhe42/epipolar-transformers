import numpy as np
from IPython import embed
from contextlib import ExitStack

import torch
from torch import nn
import torch.nn.functional as F

from .backbones.backbone import build_backbone
from .lifting import build_liftingnet
from .pictorial_cuda import rpsm
from .layers.body import HumanBody, compute_limb_length
from .metrics.metrics3d import EPEmean, EPEmean_gt, EPEmean_multiview_gt
from .metrics.metrics2d import JointsMSELoss, KeypointsMSESmoothLoss, MaskedMSELoss, compute_stage_loss, evaluate_normalized_mean_error, calculate_err, calc_pck, JDR
from core import cfg
from vision.affine_utils import normalize_points, denormalize_points
from data.transforms.build import totensor
from core.paths_catalog import BackboneCatalog
from modeling.sync_batchnorm.batchnorm import patch_sync_batchnorm, convert_model
from modeling.backbones.basic_batch import get_max_preds
from data.datasets.multiview_h36m import MultiViewH36M
from modeling import registry
from utils.checkpoint import Checkpointer

class Modelbuilder(nn.Module):
    """
    Main class for 3d keypoints
    """
    def __init__(self, cfg):
        """ construct the model here
        """
        super(Modelbuilder, self).__init__()
        if cfg.DATASETS.TASK in ['multiview_keypoint']:
            self.reference = registry.BACKBONES[cfg.BACKBONE.BODY](cfg)
            if cfg.EPIPOLAR.PRETRAINED or not cfg.EPIPOLAR.SHARE_WEIGHTS:
                backbone, backbone_dir = BackboneCatalog.get(cfg.BACKBONE.BODY)
            if cfg.EPIPOLAR.PRETRAINED:
                checkpointer = Checkpointer(
                        model=self.reference, 
                        save_dir=backbone_dir,
                )
                checkpointer.load('model.pth', prefix='backbone.module.')            
            if not cfg.VIS.FLOPS:
                self.reference = nn.DataParallel(self.reference)
            if cfg.EPIPOLAR.SHARE_WEIGHTS:
                self.backbone = self.reference
            else:
                self.backbone = registry.BACKBONES[backbone](cfg)
                # load fixed weights for the other views
                checkpointer = Checkpointer(
                        model=self.backbone, 
                        save_dir=backbone_dir,
                )
                checkpointer.load('model.pth', prefix='backbone.module.')
                self.backbone = nn.DataParallel(self.backbone)           
            if cfg.BACKBONE.SYNC_BN:
                self.reference = convert_model(self.reference)
                self.backbone = convert_model(self.backbone)
            if cfg.KEYPOINT.LOSS == 'joint':
                # if cfg.WEIGHTS != "":
                #     checkpointer = Checkpointer(model=self.reference)
                #     checkpointer.load(cfg.WEIGHTS, prefix='backbone.module.', prefix_replace='reference.module.')
                # self.reference = nn.DataParallel(self.reference)
                print('h36m special setting: JointsMSELoss')
                self.criterion = JointsMSELoss()
            elif cfg.KEYPOINT.LOSS == 'smoothmse':
                print('h36m special setting: smoothMSE')
                self.criterion = KeypointsMSESmoothLoss()
            elif cfg.KEYPOINT.LOSS == 'mse':
                _criterion = MaskedMSELoss()
                self.criterion = lambda targets, outputs: compute_stage_loss(_criterion, targets, outputs)
        elif cfg.DATASETS.TASK == 'keypoint':
            self.backbone = build_backbone(cfg)
            self.backbone = nn.DataParallel(self.backbone)
            if 'h36m' in cfg.OUTPUT_DIR:
                print('h36m special setting: JointsMSELoss')
                self.criterion = JointsMSELoss()
            else:
                _criterion = MaskedMSELoss()
                self.criterion = lambda targets, outputs: compute_stage_loss(_criterion, targets, outputs)
        elif cfg.DATASETS.TASK == 'keypoint_lifting_rot':
            self.backbone = build_backbone(cfg)
            self.backbone = nn.DataParallel(self.backbone)
            self.liftingnet = build_liftingnet()
        elif cfg.DATASETS.TASK == 'keypoint_lifting_direct':
            self.backbone = build_backbone(cfg)
            backbone, backbone_dir = BackboneCatalog.get(cfg.BACKBONE.BODY)
            checkpointer = Checkpointer(
                    model=self.backbone, 
                    save_dir=backbone_dir,
            )
            checkpointer.load('model.pth', prefix='backbone.module.')            
            self.backbone = nn.DataParallel(self.backbone)
            self.liftingnet = build_liftingnet()
            self.liftingnet = nn.DataParallel(self.liftingnet)
        elif cfg.DATASETS.TASK == 'img_lifting_rot':
            self.backbone = build_backbone(cfg)
            self.liftingnet = build_liftingnet(in_channels=self.backbone.out_channels)
            self.backbone = nn.DataParallel(self.backbone)
            self.liftingnet = nn.DataParallel(self.liftingnet)

        elif cfg.DATASETS.TASK == 'multiview_img_lifting_rot':
            self.reference= registry.BACKBONES[cfg.BACKBONE.BODY](cfg)
            backbone, backbone_dir = BackboneCatalog.get(cfg.BACKBONE.BODY)
            if True:
                checkpointer = Checkpointer(
                        model=self.reference, 
                        save_dir=backbone_dir,
                )
                checkpointer.load('model.pth', prefix='backbone.module.')
            self.reference = nn.DataParallel(self.reference)
            if cfg.EPIPOLAR.SHARE_WEIGHTS:
                self.backbone = self.reference
            else:
                self.backbone = registry.BACKBONES[backbone](cfg)
                # load fixed weights for the other views
                checkpointer = Checkpointer(
                        model=self.backbone, 
                        save_dir=backbone_dir,
                )
                checkpointer.load('model.pth', prefix='backbone.module.')
                self.backbone = nn.DataParallel(self.backbone)

            self.liftingnet = build_liftingnet(in_channels=self.backbone.out_channels)
            self.liftingnet = nn.DataParallel(self.liftingnet)
            
        elif cfg.BACKBONE.ENABLED:
            self.backbone = build_backbone(cfg)
            self.liftingnet = build_liftingnet(in_channels=self.backbone.out_channels)
            self.backbone = nn.DataParallel(self.backbone)
            self.liftingnet = nn.DataParallel(self.liftingnet)
        elif cfg.LIFTING.ENABLED:
            self.liftingnet = build_liftingnet()
            self.liftingnet = nn.DataParallel(self.liftingnet)
        else:
            raise NotImplementedError

        if cfg.KEYPOINT.TRIANGULATION == 'rpsm' and 'h36m' in cfg.OUTPUT_DIR and cfg.DATASETS.TASK in ['keypoint', 'multiview_keypoint']:
            import pickle
            self.device = torch.device('cuda:0')
            pairwise_file = cfg.PICT_STRUCT.PAIRWISE_FILE
            with open(pairwise_file, 'rb') as f:
                self.pairwise = pickle.load(f)['pairwise_constrain']
            for k, v in self.pairwise.items():
                self.pairwise[k] = torch.as_tensor(
                    v.todense().astype(np.float), device=self.device, 
                    dtype=torch.float)

    def forward(self, inputs, is_train=True):
        """ 
        To implement new tasks, put things in loss_dict, metric_dict, out
        Arguments:
            inputs (list[Tensor] ): to be processed
        Returns:
            During training, 
                it returns two dict[Tensor] for losses/metrics
        """
        loss_dict = {}
        metric_dict = {}

        if not is_train and cfg.VIS.MULTIVIEW:
            for i in inputs:
                if torch.is_tensor(inputs[i]):
                    if inputs[i].shape[0] == 1:
                        inputs[i] = inputs[i][0] #.squeeze()
        scoremap = inputs.get('heatmap')
        if scoremap is not None:
            scoremap = scoremap.to(torch.float32)    
        points2d = inputs.get('points-2d') # 333, 512
        hand_side = inputs.get('hand-side')
        img = inputs.get('img')
        target = inputs.get('can-points-3d')
        coord_xyz_rel_normed = inputs.get('normed-points-3d')
        target_global = inputs.get('points-3d')
        #print('2', target_global)
        rot_mat =  inputs.get('rotation')
        R_global = inputs.get('R')
        keypoint_scale = inputs.get('scale')
        keypoint_vis = inputs.get('visibility')
        if keypoint_vis is not None:
            keypoint_vis = keypoint_vis.to(torch.float32)    
        unit = inputs.get('unit')
        KRT = inputs.get('KRT')
        if KRT is not None:
            KRT = KRT.to(torch.float32)    
        K = inputs.get('K')
        if K is not None:
            K = K.to(torch.float32)    
        RT = inputs.get('RT')
        if RT is not None:
            RT = RT.to(torch.float32)    
        other_img = inputs.get('other_img')
        other_KRT = inputs.get('other_KRT')
        if other_KRT is not None:
            other_KRT = other_KRT.to(torch.float32)        
        other_heatmaps = inputs.get('other_heatmaps')
        quantized_img = inputs.get('quantized_img')
        other_quantized_img = inputs.get('other_quantized_img')
        camera = inputs.get('camera')
        other_camera = inputs.get('other_camera')
        crop_center = inputs.get('crop_center')
        crop_scale = inputs.get('crop_scale')
        origK = inputs.get('origK')
        if origK is not None:
            origK = origK.to(torch.float32)
        action = inputs.get('action') if 'h36m' in cfg.OUTPUT_DIR else None
        img_path = inputs.get('img-path')

        ########################################keypoint#################################################
        if cfg.DATASETS.TASK in ['keypoint', 'multiview_keypoint']:
            corr_pos = None
            if cfg.DATASETS.TASK == 'multiview_keypoint':
                if cfg.EPIPOLAR.MULTITEST:
                    with torch.no_grad():
                        #TODO support all_heatmaps
                        # all_heatmaps, all_locs, all_scos = [], [], []
                        all_locs, all_scos = [], []
                        for other_img_i, other_KRT_i, in zip(other_img, other_KRT):
                            other_features, _, _, _, _, _, _, _ = self.backbone(other_img_i)
                            #  [Nx21x224x224], Nx21x2, Nx21
                            _, batch_heatmaps, batch_locs, batch_scos, corr_pos, depths, sample_locs, warpedheatmap = self.reference(img, 
                                [other_features, other_KRT_i, None, KRT, None, None, other_img_i])
                            # batch_locs: 333 x 512
                            #all_heatmaps.append(batch_heatmaps)
                            all_locs.append(batch_locs)
                            all_scos.append(batch_scos)
                        # O x N x 21 x 224 x 224
                        #all_heatmaps = torch.stack(all_heatmaps)
                        # O x N x 21 x 2
                        all_locs = torch.stack(all_locs)
                        # O x N x 21
                        all_scos = torch.stack(all_scos)
                        #Nx21, Nx21
                        batch_scos, batch_scos_idx = torch.max(all_scos, 0)
                        batch_locs = torch.gather(all_locs, 0, 
                            batch_scos_idx[None, ..., None].expand((-1, -1, -1, 2)))
                        batch_locs = batch_locs.squeeze()
                        #batch_heatmaps = torch.gather(all_heatmaps, 0, 
                        #    batch_scos_idx[None, ..., None, None].expand((-1, -1, -1, batch_heatmaps.shape[-2], batch_heatmaps.shape[-1])))
                else:
                    with ExitStack() as stack:
                        if not cfg.EPIPOLAR.OTHER_GRAD:
                            stack.enter_context(torch.no_grad())
                        other_features, _, _, _, _, _, _, _ = self.backbone(other_img)
                    #  [Nx3x224x224], Nx21x2, Nx21
                    _, batch_heatmaps, batch_locs, batch_scos, corr_pos, depths, sample_locs, warpedheatmap = \
                        self.reference(img, [other_features, other_KRT, other_heatmaps, KRT, camera, other_camera, other_img])
            elif cfg.DATASETS.TASK == 'keypoint':
                _, batch_heatmaps, batch_locs, batch_scos, corr_pos, depths, sample_locs, warpedheatmap = self.backbone(img)

            if scoremap is not None:
                if 'h36m' in cfg.OUTPUT_DIR:
                    if is_train:
                        loss_dict['stage_loss0'] = self.criterion(batch_heatmaps[0], scoremap, keypoint_vis)
                else:
                    _, stage_loss = self.criterion(scoremap, batch_heatmaps)
                    for i, l in enumerate(stage_loss):
                        loss_dict['stage_loss' + str(i)] = l
                    if warpedheatmap is not None:
                        _, warped_stage_loss = self.criterion(scoremap, [warpedheatmap])
                        for i, l in enumerate(warped_stage_loss):
                            loss_dict['warped_loss' + str(i)] = l

            if 'h36m' in cfg.OUTPUT_DIR:
                if batch_locs is None:
                    batch_locs, batch_scos = get_max_preds(batch_heatmaps[0].detach().cpu().numpy())
                    batch_scos = batch_scos.squeeze()
                if cfg.DATASETS.H36M.MAPPING:
                    actualjoints = np.array([0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 9 , 11, 12, 14, 15, 16, 17, 18, 19])
                    batch_heatmaps[0] = batch_heatmaps[0][:, actualjoints]
                    scoremap = scoremap[:, actualjoints]
                    batch_locs = batch_locs[:, actualjoints]
                    batch_scos  = batch_scos[:, actualjoints]
                    keypoint_vis = keypoint_vis[:, actualjoints]
            if not is_train and cfg.VIS.MULTIVIEW and batch_locs is not None:
                from vision.triangulation import triangulate, triangulate_pymvg, triangulate_refine, triangulate_epipolar
                # sanity check passed!
                #batch_locs = points2d[...,:2].squeeze()
                #batch_scos = torch.ones_like(batch_scos)
                if cfg.KEYPOINT.TRIANGULATION == 'naive':
                    global_pred = triangulate(batch_locs * cfg.DATASETS.IMAGE_RESIZE * cfg.DATASETS.PREDICT_RESIZE, # 512 -> 4k
                        KRT.cpu(), 
                        batch_scos) 
                elif cfg.KEYPOINT.TRIANGULATION == 'pymvg':
                    global_pred = triangulate_pymvg(batch_locs * cfg.DATASETS.IMAGE_RESIZE * cfg.DATASETS.PREDICT_RESIZE, # 512 -> 4k
                        K.cpu().numpy(),
                        RT.cpu().numpy(), 
                        batch_scos) 
                elif cfg.KEYPOINT.TRIANGULATION == 'refine':
                    global_pred = triangulate_refine(batch_locs * cfg.DATASETS.IMAGE_RESIZE * cfg.DATASETS.PREDICT_RESIZE, # 512 -> 4k
                        KRT.cpu(), 
                        K.cpu().numpy(),
                        RT.cpu().numpy(), 
                        batch_scos) 
                elif cfg.KEYPOINT.TRIANGULATION == 'epipolar':
                    global_pred = triangulate_epipolar(batch_locs * cfg.DATASETS.IMAGE_RESIZE * cfg.DATASETS.PREDICT_RESIZE, # 512 -> 4k
                        KRT.cpu(), 
                        K.cpu().numpy(),
                        RT.cpu().numpy(), 
                        batch_scos, 
                        corr_pos, 
                        other_KRT)
                elif cfg.KEYPOINT.TRIANGULATION == 'epipolar_dlt':
                    global_pred = triangulate_epipolar(batch_locs * cfg.DATASETS.IMAGE_RESIZE * cfg.DATASETS.PREDICT_RESIZE, # 512 -> 4k
                        KRT.cpu(), 
                        K.cpu().numpy(),
                        RT.cpu().numpy(), 
                        batch_scos, 
                        corr_pos, 
                        other_KRT,
                        dlt=True)
                elif cfg.KEYPOINT.TRIANGULATION == 'rpsm':
                    boxes = []
                    poses =  target_global.cpu().numpy()
                    cameras = origK @ RT
                    for scale, center in zip(crop_scale.cpu().numpy(), crop_center.cpu().numpy()):
                        # camera_to_world_frame(datum['joints_3d'], camera['R'], camera['T'])
                        box = {}
                        box['scale'] = np.array(scale)
                        box['center'] = np.array(center)
                        boxes.append(box)
                    body = HumanBody()
                    heatmaps = batch_heatmaps[0].to(self.device)
                    kw = {
                        'body': body,
                        'boxes': boxes,
                        'center': poses[0][0],
                        'pairwise': self.pairwise,
                        'limb_length': compute_limb_length(body, poses[0])
                    }
                    # print(heatmaps)
                    # print(cameras)
                    # print(kw)
                    global_pred = rpsm(cameras, heatmaps, kw)
                else:
                    raise NotImplementedError
                global_pred = totensor(global_pred).to('cuda')
            
                EPEmean_global, EPEmean_perjoint = EPEmean(global_pred, target_global, keypoint_vis, unit=unit)
                # print(EPEmean_global)
                ## per joint confidence + EPEmean
                # conf_err = np.hstack((batch_scos.cpu().numpy().T, EPEmean_perjoint.cpu().numpy().squeeze()[:, None]))
                # print(conf_err)
                # print(conf_err[conf_err[:, -1].argsort()])
                if not torch.isnan(EPEmean_global).sum():
                    metric_dict['EPEmean_global'] = EPEmean_global
                    if 'h36m' in cfg.OUTPUT_DIR:
                        assert cfg.TEST.IMS_PER_BATCH == 1
                        for i in action[1:]:
                            assert action[0] == i
                        metric_dict['MPJPE@'+ MultiViewH36M.index_to_action_names()[int(action[0].item())]] = EPEmean_global

            # metric_dict['NMS'], metric_dict['acc@0.008'], _ = evaluate_normalized_mean_error(
            #         real_locs.transpose(-1,-2).cpu().numpy()[...,[1,0,2],:], 
            #                 points2d.transpose(-1,-2).cpu().numpy(),
            #                 keypoint_vis)

            out = {
                'heatmap_pred': batch_heatmaps[-1],
                'corr_pos'    : corr_pos,
                'depth'  : depths,
                }
            #print(batch_locs)
            #print(batch_heatmaps[-1].max())
            if batch_locs is not None:
                out['batch_locs'] = batch_locs.cpu() if torch.is_tensor(batch_locs) else batch_locs
            if batch_scos is not None:
                out['score_pred'] = batch_scos.cpu() if torch.is_tensor(batch_scos) else batch_scos
            if sample_locs is not None:
                out['sample_locs'] = sample_locs

            PCKs, err_joints, total_joints = None, None, None
            if not is_train and cfg.TEST.PCK:
                if 'h36m' in cfg.OUTPUT_DIR:
                    jdr_perjoint, metric_dict['JDR'], _, _ = JDR(
                        batch_heatmaps[0].detach().cpu().numpy(),
                        scoremap.detach().cpu().numpy())
                    for i, joint_jdr in enumerate(jdr_perjoint.squeeze()[1:]):
                        metric_dict['JDR@'+ MultiViewH36M.actual_joints[i]] = joint_jdr

                # else:
                thresholds = cfg.TEST.THRESHOLDS
                max_threshold = cfg.TEST.MAX_TH
                if cfg.DOTEST:
                    PCKs, err_joints, total_joints = calculate_err(
                        batch_locs[...,:2].transpose(-1,-2).cpu().numpy(), # 512 x 336
                        points2d[...,:2].transpose(-1,-2).cpu().numpy(), # 512 x 336
                        keypoint_vis.cpu().numpy(), 
                        thresholds,
                        max_threshold)
                    out['err_joints'] = err_joints
                    out['total_joints'] = total_joints
                else:
                    PCKs = calc_pck(batch_locs[...,:2].transpose(-1,-2).cpu().numpy(), # 512 x 336
                        points2d[...,:2].transpose(-1,-2).cpu().numpy(), # 512 x 336
                        keypoint_vis.cpu().numpy(), 
                        thresholds)
                    
                for th in thresholds:
                    metric_dict['PCK@'+str(th)] = PCKs['PCK@'+str(th)]
                    # for j in range(points2d.shape[1]):
                    #     if 'PCK@'+str(th)+'_joint_'+str(j) in PCK_joint:
                    #         metric_dict['PCK@'+str(th)+'_joint_'+str(j)] = PCK_joint['PCK@'+str(th)+'_joint_'+str(j)]


            
        ########################################lifting#################################################
        elif cfg.LIFTING.ENABLED:
            batch_size = scoremap.shape[0]
            if not is_train and cfg.VIS.MULTIVIEW:
                keypoint_scale = keypoint_scale[:, None]
                target_global = target_global[0]
            if cfg.DATASETS.TASK in ['lifting_direct', 'keypoint_lifting_direct']:
                target = coord_xyz_rel_normed
            if cfg.BACKBONE.ENABLED:
                if cfg.DATASETS.TASK == 'multiview_img_lifting_rot':
                    with torch.no_grad():
                        other_features, _, _, _ = self.backbone(other_img)
                    #  [Nx3x224x224], Nx21x2, Nx21
                    _, feat, batch_locs, batch_scos = self.reference(img, [other_features, other_KRT, KRT])
                if cfg.DATASETS.TASK == 'keypoint_lifting_direct':
                    _, feat, _, _, _, _, _, _ = self.backbone(img)
                    feat = feat[-1]
                else:
                    if cfg.BACKBONE.BODY == 'HG':
                        self.backbone.eval()
                        with torch.no_grad():
                            _, feat, _, _ = self.backbone(img)
                            feat = feat[-1]
                    else:
                        feat = self.backbone(img)
            else:
                feat = scoremap
            coord_xyz_can, R, normed_pred, global_pred = self.liftingnet(feat, hand_side, R_global)
            out = {
                    'can_pred': coord_xyz_can, 
                    'R_pred': R, 
                    'normed_pred': normed_pred, 
                    'global_pred': global_pred
                    }

            keypoint_vis = keypoint_vis.squeeze()
            vis_can = torch.mul(coord_xyz_can, keypoint_vis[..., None]).float()
            vis_tar = torch.mul(target       , keypoint_vis[..., None]).float()
            if cfg.LIFTING.AVELOSS_KP:
                criterion = torch.nn.MSELoss(reduction='mean').cuda()
                loss = criterion(vis_can, vis_tar)
            else:
                criterion = torch.nn.MSELoss(reduction='sum').cuda()
                loss = criterion(vis_can, vis_tar) / batch_size
            loss_dict['xyz_loss'] = loss
            metric_dict['EPEmean_can'] , _= EPEmean(coord_xyz_can, target, keypoint_vis, keypoint_scale, unit)

            if 'lifting_rot' in cfg.DATASETS.TASK:
                assert cfg.LIFTING.AVELOSS_KP
                rot_loss = criterion(R, rot_mat)
                loss_dict['rot_loss'] = rot_loss

                metric_dict['EPEmean'] , _= EPEmean(normed_pred, coord_xyz_rel_normed, keypoint_vis, keypoint_scale, unit)

                if not is_train and cfg.VIS.MULTIVIEW:
                    # 21 x 3
                    target_global = target_global - target_global[0]
                    global_pred = (global_pred - global_pred[0]) * keypoint_scale[..., None]
                    if cfg.LIFTING.MULTIVIEW_UPPERBOUND:
                        if target.shape[1] < 100:
                            metric_dict['EPEmean_global'] = EPEmean_multiview_gt(global_pred, 
                                    target_global, 
                                    keypoint_vis, unit=unit)
                    else:
                        if cfg.LIFTING.MULTIVIEW_MEDIUM:
                            global_pred, _ = global_pred.median(0)
                        else:
                            global_pred = global_pred.mean(0)
                        if target.shape[1] < 100:
                            metric_dict['EPEmean_global'] , _= EPEmean(global_pred, target_global, keypoint_vis[0], unit=unit)
        
        if len(loss_dict) > 1:
            # sum all loss to get total loss
            losses = sum(loss for loss in loss_dict.values())
            loss_dict['loss'] = losses
        elif len(loss_dict) == 1:
            # change the loss name to 'loss'
            loss_dict['loss'] = loss_dict.popitem()[1]
        # else:
        #     raise

        if is_train:
            return loss_dict, metric_dict
        for k in list(out.keys()):
            if out[k] is None:
                out.pop(k)
        return loss_dict, metric_dict, out
            
