import torch
import torchvision.ops
from torch import nn
from models.modules.flow_warp import flow_warp

class RefineFlow(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(RefineFlow, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.refine_flow_conv = nn.Sequential(
            nn.Conv2d(4 * in_channels + 2 * 2, self.out_channels, 1, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, 2 * self.out_channels, [4, 4], 2, [1, 1]),
            nn.ReLU(),
            nn.Conv2d(2 * self.out_channels, 4 * self.out_channels, [4, 4], 2, [1, 1]),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(4 * self.out_channels, 2 * self.out_channels, [3, 3], 1, [1, 1]),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(2 * self.out_channels, self.out_channels, [3, 3], 1, [1, 1]),
            nn.ReLU(),
            nn.Conv2d(self.out_channels, 4, [3, 3], 1, [1, 1]),
        )
    def forward(self, L1_fea, flow_backward, flow_forward, isLong = False):

        B, N, C, H, W = L1_fea.size()
        term = 1
        if isLong:
            term = 2

        for i in range(N - term):
            for_fea = L1_fea[:, i, :, :, :]
            back_fea = L1_fea[:, i+term, :, :, :]
            flow_back = flow_backward[:, i * 2 + 1, :, :, :]
            flow_for = flow_forward[:, i * 2, :, :, :]
            back_warp_fea = flow_warp(back_fea, flow_back.permute(0, 2, 3, 1))
            for_warp_fea = flow_warp(for_fea, flow_for.permute(0, 2, 3, 1))

            flow_refine_in = torch.concat([back_fea, for_fea, back_warp_fea, for_warp_fea, flow_back, flow_for], dim=1)
            flow_refine = self.refine_flow_conv(flow_refine_in)
            flow_backward[:, i * 2 + 1, :, :, :] += flow_refine[:, :2, :, :]
            flow_forward[:, i*2, :, :, :] += flow_refine[:, 2:4:, :, :]

        return flow_backward, flow_forward