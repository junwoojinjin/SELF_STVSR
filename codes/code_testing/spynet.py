import torch
import numpy as np
import cv2
import getopt
import math
import numpy
import PIL.Image
import sys

def backwarp(tenInput, tenFlow):
	backwarp_tenGrid = {}
	if str(tenFlow.shape) not in backwarp_tenGrid:
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

		backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
	# end

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
# end

##########################################################

class Preprocess(torch.nn.Module):
    def __init__(self):
        super().__init__()
    # end

    def forward(self, tenInput):
        tenInput = tenInput.flip([1])
        tenInput = tenInput - torch.tensor(data=[0.485, 0.456, 0.406], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)
        tenInput = tenInput * torch.tensor(data=[1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225], dtype=tenInput.dtype, device=tenInput.device).view(1, 3, 1, 1)

        return tenInput
    # end
# end

class Basic(torch.nn.Module):
    def __init__(self, intLevel):
        super().__init__()

        self.netBasic = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
        )
    # end

    def forward(self, tenInput):
        return self.netBasic(tenInput)
    # end
# end

class Network(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.netPreprocess = Preprocess()
		self.netBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])
		self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-spynet/network-' + arguments_strModel + '.pytorch' ).items() })
	# end

	def forward(self, ref, supp):
		tenFlow = []

		ref = [ self.netPreprocess(ref) ]
		supp = [ self.netPreprocess(supp) ]

		for intLevel in range(5):
			if ref[0].shape[2] > 32 or ref[0].shape[3] > 32:
				ref.insert(0, torch.nn.functional.avg_pool2d(input=ref[0], kernel_size=2, stride=2, count_include_pad=False))
				supp.insert(0, torch.nn.functional.avg_pool2d(input=supp[0], kernel_size=2, stride=2, count_include_pad=False))
			# end
		# end

		tenFlow = ref[0].new_zeros([ ref[0].shape[0], 2, int(math.floor(ref[0].shape[2] / 2.0)), int(math.floor(ref[0].shape[3] / 2.0)) ])

		for intLevel in range(len(ref)):
			tenUpsampled = torch.nn.functional.interpolate(input=tenFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

			if tenUpsampled.shape[2] != ref[intLevel].shape[2]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
			if tenUpsampled.shape[3] != ref[intLevel].shape[3]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

			tenFlow = self.netBasic[intLevel](torch.cat([ ref[intLevel], backwarp(tenInput=supp[intLevel], tenFlow=tenUpsampled), tenUpsampled ], 1)) + tenUpsampled
		# end

		return tenFlow
	# end
# end

##########################################################

class SPyNet(torch.nn.Module):
	"""Compute flow from ref to supp.
		        Note that in this function, the images are already resized to a
		        multiple of 32.
		        Args:
		            ref (Tensor): Reference image with shape of (n, 3, h, w).
		            supp (Tensor): Supporting image with shape of (n, 3, h, w).
		        Returns:
		            Tensor: Estimated optical flow: (n, 2, h, w).
		"""

	def __init__(self):
		super().__init__()
		self.netNetwork = Network()
	def forward(self, ref, supp):
		N , C , H, W = ref.size()
		print(ref.size())
		# en
		assert(ref.shape[1] == supp.shape[1])
		assert(ref.shape[2] == supp.shape[2])
		intWidth = W
		intHeight = H

		tenPreprocessedOne = ref.view(N, 3, intHeight, intWidth)
		tenPreprocessedTwo = supp.view(N, 3, intHeight, intWidth)

		intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 32.0) * 32.0))
		intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 32.0) * 32.0))

		tenPreprocessedOne = torch.nn.functional.interpolate(input=tenPreprocessedOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
		tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenPreprocessedTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

		tenFlow = torch.nn.functional.interpolate(input=self.netNetwork(tenPreprocessedOne, tenPreprocessedTwo), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

		tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
		tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

		return tenFlow[:, :, :, :]
# end

##########################################################


##########################################################

#assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'sintel-final' # 'sintel-final', or 'sintel-clean', or 'chairs-final', or 'chairs-clean', or 'kitti-final'
arguments_strOne = './im2.png'
arguments_strTwo = './im3.png'
arguments_strOut = './out.flo'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use, see below
	if strOption == '--one' and strArgument != '': arguments_strOne = strArgument # path to the first frame
	if strOption == '--two' and strArgument != '': arguments_strTwo = strArgument # path to the second frame
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################


if __name__ == '__main__':
	ref = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strOne))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))).unsqueeze(0)
	supp = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(arguments_strTwo))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))).unsqueeze(0)
	ref = ref.cuda()
	supp = supp.cuda()
	spyent = SPyNet().cuda()

	print(ref.shape)
	flow = spyent(ref, supp)  # (n, 2, h ,w ),  n : # batch size
	flow = torch.mul(flow, 0.1)
	print(flow.shape)
	flow = flow.cpu()
	### ***testing*** visualization flow
	flow = flow[0,:,:,:]
	flow = flow.permute(1,2,0).numpy()
	print(flow.shape)

	hsv = np.zeros((256, 448, 3), dtype=np.uint8)
	hsv[..., 1] = 255
	mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
	hsv[..., 0] = ang * 180 / np.pi / 2
	hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
	bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
	cv2.imwrite('hsv_flow' + '.jpg', bgr)


