import time, math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from core import cfg

class LiftingNet(nn.Module):
    def __init__(self, in_channels=0, DEBUG=False):
        super(LiftingNet, self).__init__()
        out_chan_list = [32, 64, 128]
        self.DEBUG = DEBUG
        if cfg.DATASETS.TASK in ['img_lifting_rot']:
            #use backbone feature
            # 3D Hand Shape and Pose from Images in the Wild
            self.poseprior = nn.Linear(in_channels + (0 if 'h36m' in cfg.OUTPUT_DIR else 2), cfg.KEYPOINT.NUM_PTS * 3)
            self.viewpoint = nn.Linear(in_channels + (0 if 'h36m' in cfg.OUTPUT_DIR else 2), 3)
        else:
            # build feature for heatmap to 3D joints
            if cfg.DATASETS.TASK in ['keypoint_lifting_rot', 'multiview_img_lifting_rot']:
                self.avgpool = torch.nn.AvgPool2d(2, stride=2, padding=0, ceil_mode=True)
            else:
                self.avgpool = torch.nn.AvgPool2d(8, stride=8, padding=0, ceil_mode=True)
            self.conv1 = nn.Sequential(
                #28x28
                nn.Conv2d(cfg.KEYPOINT.NUM_PTS              , out_chan_list[0], kernel_size=3, padding=1, stride=1),
                nn.LeakyReLU(),
                nn.Conv2d(out_chan_list[0], out_chan_list[0], kernel_size=3, padding=1, stride=2),
                #14x14
                nn.LeakyReLU(),
                nn.Conv2d(out_chan_list[0], out_chan_list[1], kernel_size=3, padding=1, stride=1),
                nn.LeakyReLU(),
                nn.Conv2d(out_chan_list[1], out_chan_list[1], kernel_size=3, padding=1, stride=2),
                #7x7
                nn.LeakyReLU(),
                nn.Conv2d(out_chan_list[1], out_chan_list[2], kernel_size=3, padding=1, stride=1),
                nn.LeakyReLU(),
                nn.Conv2d(out_chan_list[2], out_chan_list[2], kernel_size=3, padding=1, stride=2),
                #4x4
                nn.LeakyReLU(),
            )
            out_chan_list = [512, 512]

            if cfg.KEYPOINT.HEATMAP_SIZE == (64, 64):
                poseprior_input = 128
            else:
                poseprior_input = 4*4*128
            self.poseprior = nn.Sequential(
                nn.Linear(poseprior_input + (0 if 'h36m' in cfg.OUTPUT_DIR else 2), 
                    out_chan_list[0]), 
                nn.LeakyReLU(),
                nn.Dropout(1 - 0.8),
                nn.Linear(out_chan_list[0], out_chan_list[1]), 
                nn.LeakyReLU(),
                nn.Dropout(1 - 0.8),
                nn.Linear(out_chan_list[1], cfg.KEYPOINT.NUM_PTS * 3), 
            )
            if cfg.DATASETS.TASK not in ['lifting', 'lifting_direct', 'keypoint_lifting_direct']:
                out_chan_list = [64, 128, 256]
                self.conv2 = nn.Sequential(
                    #28x28
                    nn.Conv2d(cfg.KEYPOINT.NUM_PTS              , out_chan_list[0], kernel_size=3, padding=1, stride=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(out_chan_list[0], out_chan_list[0], kernel_size=3, padding=1, stride=2),
                    #14x14
                    nn.LeakyReLU(),
                    nn.Conv2d(out_chan_list[0], out_chan_list[1], kernel_size=3, padding=1, stride=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(out_chan_list[1], out_chan_list[1], kernel_size=3, padding=1, stride=2),
                    #7x7
                    nn.LeakyReLU(),
                    nn.Conv2d(out_chan_list[1], out_chan_list[2], kernel_size=3, padding=1, stride=1),
                    nn.LeakyReLU(),
                    nn.Conv2d(out_chan_list[2], out_chan_list[2], kernel_size=3, padding=1, stride=2),
                    #4x4
                    nn.LeakyReLU(),
                )
                out_chan_list = [256, 128]
                self.viewpoint = nn.Sequential(
                    nn.Linear(4*4*256 + (0 if 'h36m' in cfg.OUTPUT_DIR else 2), 
                        out_chan_list[0]), 
                    nn.LeakyReLU(),
                    nn.Dropout(1 - 0.75),
                    nn.Linear(out_chan_list[0], out_chan_list[1]), 
                    nn.LeakyReLU(),
                    nn.Dropout(1 - 0.75),
                    nn.Linear(out_chan_list[1], 3), 
                )
            

    def forward(self, x, hand_side, R_global, *argv):
        if not self.training and cfg.VIS.MULTIVIEW:
            batch_size = x.shape[0]
            #x = x.flatten(0, 1)
        batch_size = x.shape[0]
        if hand_side is not None:
            side = hand_side.type_as(x).view(-1,1)
        # extract feature
        if cfg.DATASETS.TASK not in ['img_lifting_rot', 'multiview_img_lifting_rot']:
            x = self.avgpool(x)
            if torch.isnan(x).any():
                print('x nan!')
            if cfg.DATASETS.TASK not in ['lifting', 'lifting_direct', 'keypoint_lifting_direct']:
                y = self.conv2(x)
            x = self.conv1(x)
            x = x.view(batch_size, -1)
        if hand_side is not None:            
            x = torch.cat((x, 1 - side, side), 1)

        coords_xyz_canonical = self.poseprior(x).view(batch_size, -1, 3)

        ## sanity check passed
        #if torch.isnan(x).any():
        #    print('x conv1 nan!')
        #if torch.isnan(y).any():
        #    print('y nan!')
        #if torch.isnan(coords_xyz_canonical).any():
        #    print('coords_xyz nan!')

        if cfg.DATASETS.TASK in ['lifting', 'lifting_direct', 'keypoint_lifting_direct']:
            return coords_xyz_canonical, None, None, None # n x J x 3
        
        if cfg.DATASETS.TASK in ['img_lifting_rot']:
            y = self.viewpoint(x)
        else:
            y = y.view(batch_size, -1)
            if hand_side is not None:            
                y = torch.cat((y, 1 - side, side), 1)
            y = self.viewpoint(y)

        trafo_matrix = self._get_rot_mat(y)
        if cfg.LIFTING.FLIP_ON:
            coord_xyz_can_flip = self._flip_right_hand(coords_xyz_canonical, side)
        else:
            coord_xyz_can_flip = coords_xyz_canonical

        coord_xyz_rel_normed = torch.matmul(coord_xyz_can_flip, trafo_matrix)
        if 'lifting_rot' in cfg.DATASETS.TASK:
            if not self.training and cfg.VIS.MULTIVIEW:
                Nviewshape = (
                        batch_size, 
                        coord_xyz_rel_normed.shape[-2], 
                        coord_xyz_rel_normed.shape[-1])
                coords_xyz_canonical = coords_xyz_canonical.view(Nviewshape)
                coord_xyz_rel_normed = coord_xyz_rel_normed.view(Nviewshape)
                trafo_matrix = trafo_matrix.view(batch_size, 3, 3)
                coord_xyz_global = self.multiview_avg(coord_xyz_rel_normed, R_global)

                return coords_xyz_canonical, trafo_matrix, coord_xyz_rel_normed, coord_xyz_global
            return coords_xyz_canonical, trafo_matrix, coord_xyz_rel_normed, None
        else:
            raise NotImplementedError

    @staticmethod
    def _get_rot_mat(y):
        theta = (y**2 + 1e-8).sum(1)**.5
        #theta = torch.norm(y, p=None, dim=1)

        # some tmp vars
        st= torch.sin(theta)
        ct= torch.cos(theta)
        one_ct= 1.0 - torch.cos(theta)

        norm_fac = (1.0 / theta).view(-1,1)
        u = y * norm_fac

        return torch.stack([
            ct+u[:,0]*u[:,0]*one_ct, 
            u[:,0]*u[:,1]*one_ct-u[:,2]*st, 
            u[:,0]*u[:,2]*one_ct+u[:,1]*st, 
            u[:,1]*u[:,0]*one_ct+u[:,2]*st, 
            ct+u[:,1]*u[:,1]*one_ct, 
            u[:,1]*u[:,2]*one_ct-u[:,0]*st,
            u[:,2]*u[:,0]*one_ct-u[:,1]*st, 
            u[:,2]*u[:,1]*one_ct+u[:,0]*st, 
            ct+u[:,2]*u[:,2]*one_ct], 1
            ).view(-1, 3,3)

        

    @staticmethod
    def _flip_right_hand(coords_xyz_canonical, side):
        coords_xyz_canonical_mirrored = torch.stack([
                coords_xyz_canonical[:, :, 0],
                coords_xyz_canonical[:, :, 1],
              -coords_xyz_canonical[:, :, 2]], 2)
        return torch.where(side.byte().view(-1,1,1), coords_xyz_canonical_mirrored, coords_xyz_canonical)

    @staticmethod
    def multiview_avg(coord_xyz_rel_normed, R):
        """
            coord_xyz_rel_normed: view x cfg.KEYPOINT.NUM_PTS x 3
            Rt: view x 3 x 3
        """
        assert len(coord_xyz_rel_normed.shape) == 3
        assert coord_xyz_rel_normed.shape[-1] == 3
        assert R.shape[-1] == 3
        assert R.shape[-2] == 3
        return coord_xyz_rel_normed.matmul(R.inverse().transpose(-2,-1)) #.mean(1)



def build_liftingnet(**kwargs):
    model = LiftingNet(**kwargs)
    return model

## sanity check passed
#if __name__ == '__main__':
#    x = torch.rand((8, 21, 28, 28))
#    hand_side = torch.randint(2, (8, 2)).type_as(x)
#    model = LiftingNet(DEBUG=True)
#    model.eval()
#    model(x, hand_side)
