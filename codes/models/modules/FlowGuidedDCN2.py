import torch
import torchvision.ops
from torch import nn

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.constant_(m.weight, 0.)
        nn.init.constant_(m.bias, 0.)

class FlowGuidedDCN2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=16,
                 bias=False):
        super(FlowGuidedDCN2, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.out_channels = out_channels

        self.offset_conv = nn.Sequential(
            nn.Conv2d(2 * in_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 2 * groups * kernel_size[0] * kernel_size[1], 3, 1, 1),
        )
        self.offset_conv.apply(init_weights)


        self.modulator_conv = nn.Sequential(
            nn.Conv2d(2 * in_channels + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 1 * groups * kernel_size[0] * kernel_size[1], 3, 1, 1),
        )
        self.modulator_conv.apply(init_weights)

        self.regular_conv = nn.Conv2d(in_channels=in_channels // groups,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x, warped_fea, target_f, flow, interpolate = False):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.7

        offset = self.offset_conv(torch.cat([target_f, warped_fea, flow], dim=1))  # .clamp(-max_offset, max_offset)
        offset = 20 * torch.tanh(offset)
        offset = offset + flow.flip(1).repeat(1, offset.size(1) // 2, 1, 1)

        modulator = torch.sigmoid(self.modulator_conv(torch.cat([target_f, warped_fea, flow], dim=1))) # * 1.0
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride
                                          )
        return x

    