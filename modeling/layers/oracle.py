import torch
from torch import nn
import torch.nn.functional as F

# naive implementation
class DepthOracle(nn.Module):
    def __init__(self, image_w=224, image_h=224):
        super(DepthOracle, self).__init__()
        self.w = image_w
        self.h = image_h


    def forward(self, feat, depth1, depth2, K1, R1, t1, K2, R2, t2, bbox1, bbox2):
        """ 
            feat           : N x 2 x C x W x H
            bbox1           : N x 4   (x1, y1, x2, y2)
            K1_inv          : N x 3 x 3
            R1_inv          : N x 3 x 3
            t1:             : N x 3 
            depth1         : N x W x H
            calculate K_inv beforehand
        """

        N = feat.shape[0]

        K1_inv = torch.inverse(K1)
        R1_inv = torch.inverse(R1)
        K2_inv = torch.inverse(K2)
        R2_inv = torch.inverse(R2)
        
        rel_feat2 = self.append_feature(feat, 0, 1, depth1, depth2, K1_inv, R1_inv, t1, K2, R2, t2, bbox1)
        rel_feat1 = self.append_feature(feat, 1, 0, depth1, depth2, K2_inv, R2_inv, t2, K1, R1, t1, bbox2)

        rel_feat = torch.cat((rel_feat2, rel_feat1), dim=1)
        assert (list(rel_feat.shape) == [N, 2, feat.shape[2], 224, 224])
        feat = torch.cat((feat, rel_feat), dim=2)
        return feat


    def append_feature(self, feat, id1, id2, depth1, depth2, K1_inv, R1_inv, t1, K2, R2, t2, bbox):
        """
            append feat[:, id2, :, :, :] onto id1
        """
        
        N = feat.shape[0]

        # get original coord mesh grid (decropped coordinates)
        dc_x, dc_y = self.decrop(self.w, self.h, bbox) # N x 224 x 224
        ones = torch.ones((N, 1, self.w, self.h))
        homo_2d = torch.cat((dc_x.unsqueeze(1), dc_y.unsqueeze(1), ones), dim=1)
        assert (list(homo_2d.shape) == [N, 3, 224, 224]), homo_2d.shape
        local_3d = homo_2d * depth1.unsqueeze(1) # [dx, dy, d]
        assert (list(homo_2d.shape) == [N, 3, 224, 224])

        # multiply the inverse of the intrinsics matrix
        local_3d = local_3d.reshape(N, 3, local_3d.shape[2] * local_3d.shape[3])
        assert (list(local_3d.shape) == [N, 3, 224*224])
        cali_local_3d = torch.matmul(K1_inv, local_3d)  # N x 3 x 3,  N x 3 x (224*224) -> N x 3 x (224*224)
        assert (list(cali_local_3d.shape) == [N, 3, 224*224])

        # remove t
        cali_local_3d = cali_local_3d - t1.unsqueeze(2)
        assert (list(cali_local_3d.shape) == [N, 3, 224*224])
    
        # global 3d
        global_3d = torch.matmul(R1_inv, cali_local_3d) # Nx3x3 @ Nx3x(224*224)
        assert (list(global_3d.shape) == [N, 3, 224*224])

        # 2d coord for the other view
        homo_global_3d = torch.cat((global_3d, torch.ones(N, 1, self.w*self.h)), dim=1)
        assert (list(homo_global_3d.shape) == [N, 4, 224*224])
        Rt2 = torch.cat((R2, t2.unsqueeze(2)), dim=2)
        rel_2d = torch.matmul(Rt2, homo_global_3d) # N x 3 x 4 @ N x 4 x (224*224)
        assert (list(rel_2d.shape) == [N, 3, 224*224])
        rel_2d = torch.matmul(K2, rel_2d)
        assert (list(rel_2d.shape) == [N, 3, 224*224])

        # check depth matches (not required)
        # assert torch.max(rel_2d[:, 2, :] - depth2.view(N,  self.w * self.h)) < 0.01

        # get local 2d on the other view
        homo_2d = rel_2d / rel_2d[:, 2, :].unsqueeze(1)
        homo_2d = homo_2d.transpose(1, 2)
        assert (list(homo_2d.shape) == [N, 224*224, 3]), homo_2d.shape
        homo_2d = homo_2d.reshape(N, self.w, self.h, 3)
        local_2d = (homo_2d / self.w)[:, :, :, :2]
        assert (list(local_2d.shape) == [N, 224, 224, 2]) 

        # the grid_sample requairs the grid mostly in [-1, 1]
        grid = (local_2d / self.w * 2) - 1
        rel_feat = F.grid_sample(feat[:, id2, :, :, :], grid).unsqueeze(1)
        assert (list(rel_feat.shape) == [N, 1, feat.shape[2], 224, 224]) 
        return rel_feat


    def decrop(self, image_w, image_h, bbox):
        """
            bbox : N x 4
            return:
                x: N x 224
                y: N x 224
        """
        W, H = 667, 1024 
        N = bbox.shape[0]

        scalex = (bbox[:, 2] - bbox[:, 0]) / image_w * W # N x 1
        scaley = (bbox[:, 3] - bbox[:, 1]) / image_h * H # N x 1

        # meshgrid before transform because I don't know how to mbatch meshgrid. It could be speed up by reshape or something
        x, y = torch.meshgrid(torch.arange(image_w), torch.arange(image_h)) # x:  224 x 224
        x = x.repeat(N, 1, 1).float() # N x 224 x 224
        y = y.repeat(N, 1, 1).float() # N x 224 x 224
        assert (list(x.shape) == [N, 224, 224]), x.shape


        x = x * scalex.unsqueeze(1).unsqueeze(2) + bbox[:, 0].unsqueeze(1).unsqueeze(2) # N x 224 x 224
        y = y * scaley.unsqueeze(1).unsqueeze(2) + bbox[:, 1].unsqueeze(1).unsqueeze(2) # N x 224 x 224

        assert (list(x.shape) == [N, 224, 224])

        return x, y

if __name__ == '__main__':
    feat = torch.rand(8, 2, 5, 224, 224)
    depth1 = torch.rand(8, 224, 224)*100
    depth2 = torch.rand(8, 224, 224)*100

    K1 = torch.Tensor([
        [5088.65689580784, 0, 762.798050644136], 
        [0, 5088.3413444753, 972.532917295082], 
        [0, 0, 1]])
    K2 = torch.Tensor([
        [5045.37873105597, 0, 620.395283981989], 
        [0, 5044.469444263, 955.685765421187], 
        [0, 0, 1]])

    K1_inv = torch.inverse(K1)
    K2_inv = torch.inverse(K2)

    Rt1 = torch.Tensor([[0.95489279623216905, 0.02735518809099674, 0.29568808124172918, -337.42068540581846], [-0.020280805465591939, 0.99943059138612278, -0.026966311042700475, 1.1393122370077697], [-0.29625738241194266, 0.019753143701473026, 0.95490385729681337, 104.01817963092948]])
    Rt2 = torch.Tensor([[0.90791772422834294, -0.0098093716533857594, -0.41903362903207714, 438.43913408369815], [-0.11108666701234909, 0.95834486082964565, -0.26312521759324881, 276.9909782735827], [0.40415981794843753, 0.28544509795961259, 0.86900859466791991, 192.3146319561659]])
    Rt1 = Rt1.repeat(8, 1, 1)
    Rt2 = Rt2.repeat(8, 1, 1)

    R1 = Rt1[:, :, :3]
    R2 = Rt2[:, :, :3]
    t1 = Rt1[:, :, 3]
    t2 = Rt2[:, :, 3]

    R1_inv, R2_inv = torch.inverse(R1), torch.inverse(R2)

    bbox = torch.rand(2, 8, 4)
    bbox[:, :, 0] = bbox[:, :, 0] * 667
    bbox[:, :, 2] = bbox[:, :, 2] * 667
    bbox[:, :, 1] = bbox[:, :, 1] * 1024
    bbox[:, :, 3] = bbox[:, :, 3] * 1024

    bbox1 = bbox[0, :, :]
    bbox2 = bbox[1, :, :]

    net = DepthOracle()
    net.eval()
    new_feat = net(feat, depth1, depth2, K1, R1, t1, K2, R2, t2, bbox1, bbox2)
    print(new_feat.shape)
    print("Yeah")
