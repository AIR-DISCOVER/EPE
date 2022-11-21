import logging
from math import sqrt

import torch
import torch.nn as nn



logger = logging.getLogger('epe.nf')

norm_factory = {\
	'none': None,
	'group': lambda d: nn.GroupNorm(8,d),
	'batch': lambda d: nn.BatchNorm2d(d, track_running_stats=False),
	'inst':  lambda d: nn.InstanceNorm2d(d, affine=True, track_running_stats=False),
	'domain':lambda d: nn.DomainNorm(d),
}


def make_conv_layer(dims, strides=1, leaky_relu=True, spectral=False, norm_factory=None, skip_final_relu=False, kernel=3):
	""" Make simple convolutional networks without downsampling.

	dims -- list with channel widths, where len(dims)-1 is the number of concolutional layers to create.
	strides -- stride of first convolution if int, else stride of each convolution, respectively
	leaky_relu -- yes or no (=use ReLU instead)
	spectral -- use spectral norm
	norm_factory -- function taking a channel width and returning a normalization layer.
	skip_final_relu -- don't use a relu at the end
	kernel -- width of kernel
	"""

	if type(strides) == int:
		strides = [strides] + [1] * (len(dims)-2)
		pass

	c = nn.Conv2d(dims[0], dims[1], kernel, stride=strides[0], bias=spectral)
	m = [] if kernel == 1 else [nn.ReplicationPad2d(kernel // 2)]
	m += [c if not spectral else torch.nn.utils.spectral_norm(c)]

	if norm_factory:
		m += [norm_factory(dims[1])]
		pass

	m += [nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)]

	num_convs = len(dims)-2
 
	for i,di in enumerate(dims[2:]):
		
		c = nn.Conv2d(dims[i+1], di, 3, stride=strides[i+1], bias=spectral, groups=8)
	
		if kernel > 1:
			m += [nn.ReplicationPad2d(kernel // 2)]
		m += [c if not spectral else torch.nn.utils.spectral_norm(c)]
		
		if norm_factory:
			m += [norm_factory(di)]
			pass
		
		if i == num_convs-1 and skip_final_relu:
			continue
		else:
			m += [nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)]
		pass

	return nn.Sequential(*m)

def channel_shuffle(x, groups=8):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)
    return x

def channel_split(features, ratio=0.5):
	"""
	ratio: c'/c, default value is 0.5
	""" 
	size = features.size()[1]
	split_idx = int(size * ratio)
	return features[:,:split_idx,:,:], features[:,split_idx:,:,:]



class ResBlock(nn.Module):
	def __init__(self, dims, first_stride=1, leaky_relu=True, spectral=False, norm_factory=None, kernel=3):
		super(ResBlock, self).__init__()
		self.conv = make_conv_layer(dims, first_stride, leaky_relu, spectral, norm_factory, True, kernel=kernel)
		self.down = make_conv_layer([dims[0], dims[-1]], first_stride, leaky_relu, spectral, None, True, kernel=kernel) \
			if first_stride != 1 or dims[0] != dims[-1] else None
		self.relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
		pass

	def forward(self, x):
		return self.relu(self.conv(x) + (x if self.down is None else self.down(x)))


class ResBlockOpt(nn.Module):
	"""
	This class aims to substitue the original ResBlock with the Shuffle Net V2 block. The parameter dims'
	length must be 2 or 3.
 	"""
	def __init__(self, dims, first_stride=1, leaky_relu=True, spectral=False, norm_factory=None, kernel=3, ratio=0.5):
		super(ResBlockOpt, self).__init__()
		self.ratio = ratio
		self.dims = dims
		if len(dims) == 2:
			assert dims[0] == dims[1], f"Dims with len 2 must satisfy dims[0] == dims[1], but now with \
   										dims[0] = {dims[0]}, dims[1] = {dims[1]}"
			self.indicate = 0
			self.conv_num = 1
			self.new_dims = [int(dims[0]*ratio), int(dims[1]*ratio)]
			self.conv1 = make_conv_layer(self.new_dims, first_stride, leaky_relu, spectral, norm_factory, True, kernel=kernel)
		elif len(dims) == 3:
			assert dims[0] == dims[1] or dims[1] == dims[2], f"Dims with len 3 must satisfy dims[0] == dims[1] \
   				or dims[1] == dims[2], but now with dims[0] = {dims[0]}, dims[1] = {dims[1]}, dims[2] = {dims[2]}"
       
			if dims[0] == dims[1] and dims[1] == dims[2]:
				self.conv_num = 1
				self.new_dims = [int(dim*ratio) for dim in dims]
				self.conv1 = make_conv_layer(self.new_dims, first_stride, leaky_relu, spectral, norm_factory, True, kernel=kernel)
			elif dims[0] == dims[1]:
				self.conv_num = 2
				self.new_dims = [int(dims[0]*ratio), int(dims[1]*ratio)]
				self.conv1 = make_conv_layer(self.new_dims, first_stride, leaky_relu, spectral, norm_factory, True, kernel=kernel)
				self.conv2 = make_conv_layer(dims[1:], first_stride, leaky_relu, spectral, norm_factory, True, kernel=kernel)
				self.indicate = 0
			else:
				self.conv_num = 2
				self.new_dims = [int(dims[1]*ratio), int(dims[2]*ratio)]
				self.conv1 = make_conv_layer(dims[:2], first_stride, leaky_relu, spectral, norm_factory, True, kernel=kernel)
				self.conv2 = make_conv_layer(self.new_dims, first_stride, leaky_relu, spectral, norm_factory, True, kernel=kernel)
				self.indicate = 1
		else:
			raise "Not Implemented Optimization"
		
		self.down_new_dims = [int(dims[0]*ratio), int(dims[1]*ratio)] if self.indicate == 0 else \
  								[int(dims[1]*ratio), int(dims[2]*ratio)]
		self.down = make_conv_layer(self.down_new_dims, first_stride, leaky_relu, spectral, \
                              None, True, kernel=kernel) if first_stride != 1 or dims[0] != dims[-1] else None
	
	def forward(self, x):
		if self.conv_num == 1:
			x1, x2 = channel_split(x, ratio=self.ratio)
			x1 = self.conv1(x1)
			if self.down is not None:
				x2 = self.down(x2)
			res = torch.cat((x1, x2), dim=1)
		else:
			if self.indicate == 0:
				x1, x2 = channel_split(x, ratio=self.ratio)
				x1 = self.conv1(x1)
				if self.down is not None:
					x2 = self.down(x2)
				res = torch.cat((x1, x2), dim=1)
				res = self.conv2(res)
			elif self.indicate == 1:
				x = self.conv1(x)
				x1, x2 = channel_split(x, ratio=self.ratio)
				if self.down is not None:
					x2 = self.down(x2)
				x1 = self.conv2(x1)
				res = torch.cat((x1, x2), dim=1)
		res = channel_shuffle(res, groups=2)
		return res



class Res2Block(nn.Module):
	def __init__(self, dims, first_stride=1, leaky_relu=True):
		super(Res2Block, self).__init__()

		self.conv = make_conv_layer(dims, first_stride, leaky_relu, False, None, False, kernel=3)
		self.down = make_conv_layer([dims[0], dims[-1]], first_stride, leaky_relu, False, None, True, kernel=1) \
			if first_stride != 1 or dims[0] != dims[-1] else None
		self.relu = nn.LeakyReLU(0.2, inplace=True) if leaky_relu else nn.ReLU(inplace=True)
		pass

	def forward(self, x):
		return 0.1 * self.conv(x) + (x if self.down is None else self.down(x))


class BottleneckBlock(nn.Module):
	def __init__(self, dim_in, dim_mid, dim_out, stride=1):
		super(BottleneckBlock, self).__init__()
		self._conv1 = nn.Conv2d(dim_in, dim_mid, 1)
		self._conv2 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(dim_mid, dim_mid, 3, stride=stride))
		self._conv3 = nn.Conv2d(dim_mid, dim_out, 1)
		self._relu  = nn.LeakyReLU(0.2, True) if leaky_relu else nn.ReLU(True)
		self._norm1 = nn.GroupNorm(dim_mid)
		self._norm2 = nn.GroupNorm(dim_mid)
		self._norm3 = nn.GroupNorm(dim_out)
		self._down  = nn.Conv2d(dim_in, dim_out, 1, stride=stride) if stride > 1 or dim_in != dim_out else None
		pass

	def forward(self, x):
		r = x if self_down is None else self._down(x)
		x = self._conv1(x)
		x = self._norm1(x)
		x = self._relu(x)
		x = self._conv2(x)
		x = self._norm2(x)
		x = self._relu(x)
		x = self._conv3(x)
		x = self._norm3(x)
		x = x + r			
		x = self._relu(x)
		return x


class ResnextBlock(nn.Module):
	def __init__(self, dim_in, dim_mid, dim_out, groups=8, stride=1):
		super(ResnextBlock, self).__init__()
		self._conv1 = nn.Conv2d(dim_in, dim_mid, 1)
		self._conv2 = nn.Sequential(nn.ReplicationPad2d(1), nn.Conv2d(dim_mid, dim_mid, 3, stride=stride, groups=groups))
		self._conv3 = nn.Conv2d(dim_mid, dim_out, 1)
		self._relu  = nn.LeakyReLU(0.2, True) if False else nn.ReLU(True)
		self._norm1 = nn.GroupNorm(groups, dim_mid)
		self._norm2 = nn.GroupNorm(groups, dim_mid)
		self._norm3 = nn.GroupNorm(groups, dim_out)
		self._down  = nn.Conv2d(dim_in, dim_out, 1, stride=stride) if stride > 1 or dim_in != dim_out else None
		pass

	def forward(self, x):
		r = x if self._down is None else self._down(x)
		x = self._conv1(x)
		x = self._norm1(x)
		x = self._relu(x)
		x = self._conv2(x)
		x = self._norm2(x)
		x = self._relu(x)
		x = self._conv3(x)
		x = self._norm3(x)
		x = x + r			
		x = self._relu(x)
		return x


