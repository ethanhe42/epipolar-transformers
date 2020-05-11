import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision.multiview import findFundamentalMat

class Meta(nn.Module):
    def __init__(self, in_channels, kernel_size=1, stride=1, padding=0):
        """
        Args:
            shape: NCHW
        """
        super(Meta, self).__init__()
        self.shape = [in_channels, in_channels, kernel_size, kernel_size]
        self.embedding_size = 9
        hidden_size = 100
        out_channels = np.prod(self.shape)
        self.stride = stride
        self.padding = padding
        self.fc0 = nn.Linear(self.embedding_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, out_channels)
        self.bias = torch.zeros(in_channels, requires_grad=True)
        self.share = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, KRT, other_KRT, input):
        """
        Args:
            KRT, other_KRT: N34
            input: N C H W
        """
        N, C, H, W = input.shape
        # N x 3 x 3
        fundamental_mat = findFundamentalMat(KRT, other_KRT)
        # N x 9 -> N x hidden_size
        hidden = self.fc0(fundamental_mat.view(-1, self.embedding_size))
        hidden = F.relu(hidden)
        weight = self.fc1(hidden)
        # N x C C kh kw 
        weight = weight.view([N, *self.shape])
        out = []
        for i in range(N):
            tmp = F.conv2d(input[i].view(1, C, H, W),
                weight[i],
                bias=self.bias.to(input),
                stride=self.stride, 
                padding=self.padding)
            out.append(tmp.squeeze())
        return torch.stack(out) + self.share(input)
        

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "size=" + str(self.shape)
        tmpstr += ")"
        return tmpstr
