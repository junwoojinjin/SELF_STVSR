''' network architecture for Sakuya '''
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.spynet import SPyNet
import models.modules.module_util as mutil
from models.modules.convlstm import ConvLSTM, ConvLSTMCell
from models.modules.flow_warp import flow_warp
from models.modules.FlowGuidedDCN2 import FlowGuidedDCN2

# from models.modules.FGDCN_2 import FlowGuidedDCN2

"""try:
    from models.modules.DCNv2.dcn_v2 import DCN_sep
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')
"""


class Real_STVR(nn.Module):
    def __init__(self, nf=64, nframes=3, groups=8, front_RBs=5, back_RBs=30, DCN_RBs=10):
        super(Real_STVR, self).__init__()
        self.nf = nf
        self.in_frames = 1 + nframes // 2
        self.ot_frames = nframes
        p_size = 48  # a place holder, not so useful
        patch_size = (p_size, p_size)
        n_layers = 1
        hidden_dim = []
        for i in range(n_layers):
            hidden_dim.append(nf)

        self.spynet = SPyNet()
        ResidualBlock_noBN_f = functools.partial(
            mutil.ResidualBlock_noBN, nf=nf)

        ResidualBlock_init_2_f = functools.partial(
            mutil.ResidualBlock_init, nf=nf, mul=2)

        ResidualBlock_init_3_f = functools.partial(
            mutil.ResidualBlock_init, nf=nf, mul=3)

        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction_VFI = mutil.make_layer(
            ResidualBlock_noBN_f, front_RBs)

        self.fgDCN_forward = FlowGuidedDCN2(nf, nf, 3, 1, 1)
        self.fgDCN_backward = FlowGuidedDCN2(nf, nf, 3, 1, 1)
        # self.fusion_inter = mutil.make_layer_diff_init(ResidualBlock_init_2_f, ResidualBlock_noBN_f, 3)
        self.fusion_inter = nn.Conv2d(2 * nf, nf, 1, 1, bias=True)

        # DCN_SR
        self.fgDCN_forward_SR = FlowGuidedDCN2(nf, nf, 3, 1, 1)
        self.fgDCN_backward_SR = FlowGuidedDCN2(nf, nf, 3, 1, 1)
        self.feature_extraction_VSR_backward = mutil.make_layer_diff_init(ResidualBlock_init_2_f, ResidualBlock_noBN_f,
                                                                          DCN_RBs)
        self.feature_extraction_VSR_forward = mutil.make_layer_diff_init(ResidualBlock_init_2_f, ResidualBlock_noBN_f,
                                                                         DCN_RBs)
        # reconstruction

        # upsampling
        self.fusion = mutil.make_layer_diff_init(ResidualBlock_init_3_f, ResidualBlock_noBN_f, back_RBs)
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.upconv2 = nn.Conv2d(nf, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle2 = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def compute_flow(self, lrs):
        B, N, C, H, W = lrs.size()

        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, C, H, W)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, C, H, W)

        flows_backward = self.spynet(lrs_1, lrs_2).view(B, N - 1, 2, H, W)
        flows_forward = self.spynet(lrs_2, lrs_1).view(B, N - 1, 2, H, W)

        return flows_forward, flows_backward  ## backward: 1 -> 2 , forward 2 -> 1

    def make_inter_flows(self, flows_backward, flows_forward):
        B, N, C, H, W = flows_backward.size()
        result_for = []
        result_back = []

        for i in range(N * 2):
            # flow from its flow
            if i % 2 == 0:
                result_back.append(torch.mul(flows_backward[:, i // 2, :, :, :], 0.5))
                result_for.append(torch.mul(flows_backward[:, i // 2, :, :, :], -0.5))

            # flow from others flow
            else:
                result_back.append(torch.mul(flows_forward[:, i // 2, :, :, :], -0.5))
                result_for.append(torch.mul(flows_forward[:, i // 2, :, :, :], 0.5))

        return torch.stack(result_back, dim=1), torch.stack(result_for, dim=1)

    def make_long_inter(self, N, L1_fea, flows_forward, flows_backward):
        long_backward_inter_fea = []
        for i in range(N - 2, 0, -1):  # 2 -> 1
            w_L1_fea = L1_fea[:, i + 1, :, :, :]
            flow = flows_backward[:, i, :, :, :]  # B, 2, H, W
            warped_fea = flow_warp(w_L1_fea, flow.permute(0, 2, 3, 1))
            inter_fea = self.fgDCN_backward(w_L1_fea, warped_fea, L1_fea[:, i - 1, :, :, :], flow, False)
            long_backward_inter_fea.append(inter_fea)

        long_backward_inter_fea = long_backward_inter_fea[::-1]

        # make forward interpolated L1 feature
        long_forward_inter_fea = []
        for i in range(1, N - 1):  # 1 -> 2
            w_L1_fea = L1_fea[:, i - 1, :, :, :]
            flow = flows_forward[:, i - 1, :, :, :]
            warped_fea = flow_warp(w_L1_fea, flow.permute(0, 2, 3, 1))
            inter_fea = self.fgDCN_forward(w_L1_fea, warped_fea, L1_fea[:, i + 1, :, :, :], flow, False)
            long_forward_inter_fea.append(inter_fea)

        # make full_interpolated LR-feature by blending
        long_fusion_inter_frames = []
        for i in range(len(long_forward_inter_fea)):
            long_fusion_inter_frames.append(
                self.fusion_inter(torch.cat([long_backward_inter_fea[i], long_forward_inter_fea[i]], dim=1)))
        return long_fusion_inter_frames

    def do_vsr(self, N, lrs, flows_forward, flows_backward):

        return 1

    def forward(self, x):
        B, N, C, H, W = x.size()  # N input video frames
        # b, 4, 3, 32,32
        # extract LR features
        L1_fea = self.lrelu(self.conv_first(x.view(-1, C, H, W)))
        L1_fea = self.feature_extraction_VFI(L1_fea)
        L1_fea = L1_fea.view(B, N, -1, H, W)

        # compute LR flow
        flows_forward, flows_backward = self.compute_flow(x)

        L1_fea_c, flows_forward_c, flows_backward_c = L1_fea.clone().detach(), flows_forward.clone().detach(), flows_backward.clone().detach()

        # make interpolated 0 4 6 for interpolation module loss
        long_fusion_inter_frames = self.make_long_inter(N, L1_fea_c, flows_forward_c, flows_backward_c)
        long_fusion_inter_frames = torch.stack(long_fusion_inter_frames, dim=1)

        # make VSR
        long_N = long_fusion_inter_frames.size()[1]
        flows_backward_c = flows_backward_c[:, 1:-1, :, :, :]
        flows_forward_c = flows_forward_c[:, 1:-1, :, :, :]
        output_backward = []

        fea_prev = L1_fea_c.new_zeros(B, self.nf, H, W)
        for i in range(long_N - 1, -1, -1):
            fea_now = long_fusion_inter_frames[:, i, :, :, :]
            if i < long_N - 1:
                flow = flows_backward_c[:, i, :, :, :]
                fea_warped = flow_warp(fea_prev, flow.permute(0, 2, 3, 1))
                fea_now = self.fgDCN_backward_SR(fea_prev, fea_warped, fea_now, flow, False)

            fea_prev = torch.cat([fea_now, fea_prev], dim=1)  # 8 , 128, 32, 32
            fea_prev = self.feature_extraction_VSR_backward(fea_prev)

            output_backward.append(fea_prev)

        output_forward_from_back_i = output_backward[::-1]

        # forward
        fea_prev = torch.zeros_like(fea_prev)
        for i in range(0, long_N):
            fea_now = long_fusion_inter_frames[:, i, :, :, :]
            if i > 0:
                flow = flows_forward_c[:, i - 1, :, :, :]
                fea_warped = flow_warp(fea_prev, flow.permute(0, 2, 3, 1))
                fea_now = self.fgDCN_forward_SR(fea_prev, fea_warped, fea_now, flow, False)

            fea_prev = torch.cat([fea_now, fea_prev], dim=1)
            fea_prev = self.feature_extraction_VSR_forward(fea_prev)

            out = torch.cat([long_fusion_inter_frames[:, i, :, :, :], output_forward_from_back_i[i], fea_prev], dim=1)
            out = self.lrelu(self.fusion(out))

            out = self.lrelu(self.upconv1(out))
            out = self.lrelu(self.pixel_shuffle1(out))

            out = self.lrelu(self.upconv2(out))
            out = self.lrelu(self.pixel_shuffle2(out))

            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            output_forward_from_back_i[i] = out

        # make interpolated flows
        flows_backward, flows_forward = self.make_inter_flows(flows_backward, flows_forward)

        # make backward interpolated L1 feature
        backward_inter_fea = []
        for i in range(N - 2, -1, -1):  # 2 -> 1 -> 0
            w_L1_fea = L1_fea[:, i + 1, :, :, :]
            flow = flows_backward[:, i * 2 + 1, :, :, :]  # B, 2, H, W
            warped_fea = flow_warp(w_L1_fea, flow.permute(0, 2, 3, 1))
            inter_fea = self.fgDCN_backward(w_L1_fea, warped_fea, L1_fea[:, i, :, :, :], flow, False)
            backward_inter_fea.append(inter_fea)

        backward_inter_fea = backward_inter_fea[::-1]
        # make forward interpolated L1 feature
        forward_inter_fea = []
        for i in range(1, N):  # 1 -> 2 -> 3
            w_L1_fea = L1_fea[:, i - 1, :, :, :]
            flow = flows_forward[:, (i - 1) * 2, :, :, :]
            warped_fea = flow_warp(w_L1_fea, flow.permute(0, 2, 3, 1))
            inter_fea = self.fgDCN_forward(w_L1_fea, warped_fea, L1_fea[:, i, :, :, :], flow, False)
            forward_inter_fea.append(inter_fea)

        # make full_interpolated LR-feature by blending
        fusion_inter_frames = []
        for i in range(len(forward_inter_fea)):
            fusion_inter_frames.append(
                self.fusion_inter(torch.cat([backward_inter_fea[i], forward_inter_fea[i]], dim=1)))
            # fusion_inter_frames.append(torch.mul(backward_inter_fea[i], 0.5) + torch.mul(forward_inter_fea[i], 0.5))

        # make all frames tensors
        lrs = []
        for i in range(N * 2 - 1):
            if i % 2 == 0:  # origin(input) LR frame
                lrs.append(L1_fea[:, i // 2, :, :, :])
            else:
                lrs.append(fusion_inter_frames[i // 2])

        lrs = torch.stack(lrs, dim=1)  # high frame rate
        B, N, C, H, W = lrs.size()

        # make interpolated flows
        # flows_backward, flows_forward = self.make_inter_flows(flows_backward, flows_forward)

        # do VSR
        # backward
        output_backward = []
        fea_prev = lrs.new_zeros(B, self.nf, H, W)
        for i in range(N - 1, -1, -1):
            fea_now = lrs[:, i, :, :, :]
            if i < N - 1:
                flow = flows_backward[:, i, :, :, :]
                fea_warped = flow_warp(fea_prev, flow.permute(0, 2, 3, 1))
                fea_now = self.fgDCN_backward_SR(fea_prev, fea_warped, fea_now, flow, False)

            fea_prev = torch.cat([fea_now, fea_prev], dim=1)  # 8 , 128, 32, 32
            fea_prev = self.feature_extraction_VSR_backward(fea_prev)

            output_backward.append(fea_prev)

        output_forward_from_back = output_backward[::-1]

        # forward
        fea_prev = torch.zeros_like(fea_prev)
        for i in range(0, N):
            fea_now = lrs[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                fea_warped = flow_warp(fea_prev, flow.permute(0, 2, 3, 1))
                fea_now = self.fgDCN_forward_SR(fea_prev, fea_warped, fea_now, flow, False)

            fea_prev = torch.cat([fea_now, fea_prev], dim=1)
            fea_prev = self.feature_extraction_VSR_forward(fea_prev)

            out = torch.cat([lrs[:, i, :, :, :], output_forward_from_back[i], fea_prev], dim=1)
            out = self.lrelu(self.fusion(out))

            out = self.lrelu(self.upconv1(out))
            out = self.lrelu(self.pixel_shuffle1(out))

            out = self.lrelu(self.upconv2(out))
            out = self.lrelu(self.pixel_shuffle2(out))

            out = self.lrelu(self.conv_hr(out))
            out = self.conv_last(out)
            output_forward_from_back[i] = out

        return torch.stack(output_forward_from_back, dim=1), torch.stack(output_forward_from_back_i, dim=1), L1_fea_c[:,
                                                                                                             1:-1, :, :,
                                                                                                             :].clone(), long_fusion_inter_frames
