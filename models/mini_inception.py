from collections import namedtuple
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Optional
from torch import Tensor

model_urls = {
	# Inception v3 ported from TensorFlow
	'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

InceptionOutputs = namedtuple('InceptionOutputs', ['logits'])
InceptionOutputs.__annotations__ = {'logits': torch.Tensor}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs


class MiniInception(nn.Module):

	def __init__(self, num_classes=1, transform_input=False, init_weights=True):
		super(MiniInception, self).__init__()

		conv_block = BasicConv2d
		self.transform_input = transform_input
		self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
		self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
		self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
		self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
		self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
		self.Mixed_5b = Inception_A(192, pool_features=32)
		self.Mixed_5c = Inception_A(256, pool_features=64)
		self.Mixed_5d = Inception_A(288, pool_features=64)
		self.Mixed_6a = Inception_Reduction_A(288)
		self.Mixed_6b = Inception_B(768, channels_7x7=128)
		self.Mixed_6c = Inception_B(768, channels_7x7=160)
		self.Mixed_6d = Inception_B(768, channels_7x7=160)
		self.Mixed_6e = Inception_B(768, channels_7x7=192)
		self.Mixed_7a = Inception_Reduction_B(768)
		self.Mixed_7b = Inception_C(1280)
		self.Mixed_7c = Inception_C(2048)
		self.fc = nn.Linear(2048, num_classes)

		if init_weights:
			for m in self.modules():
				if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
					import scipy.stats as stats
					stddev = m.stddev if hasattr(m, 'stddev') else 0.1
					X = stats.truncnorm(-2, 2, scale=stddev)
					values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
					values = values.view(m.weight.size())
					with torch.no_grad():
						m.weight.copy_(values)
				elif isinstance(m, nn.BatchNorm2d):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)

	def _transform_input(self, x):
		if self.transform_input:
			x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
			x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
			x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
			x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
		return x

	def _forward(self, x):
		# N x 3 x 299 x 299
		x = self.Conv2d_1a_3x3(x)
		# N x 32 x 149 x 149
		x = self.Conv2d_2a_3x3(x)
		# N x 32 x 147 x 147
		x = self.Conv2d_2b_3x3(x)
		# N x 64 x 147 x 147
		x = F.max_pool2d(x, kernel_size=3, stride=2)
		# N x 64 x 73 x 73
		x = self.Conv2d_3b_1x1(x)
		# N x 80 x 73 x 73
		x = self.Conv2d_4a_3x3(x)
		# N x 192 x 71 x 71
		x = F.max_pool2d(x, kernel_size=3, stride=2)
		# N x 192 x 35 x 35
		x = self.Mixed_5b(x)
		# N x 256 x 35 x 35
		x = self.Mixed_5c(x)
		# N x 288 x 35 x 35
		x = self.Mixed_5d(x)
		# N x 288 x 35 x 35
		x = self.Mixed_6a(x)
		# N x 768 x 17 x 17
		x = self.Mixed_6b(x)
		# N x 768 x 17 x 17
		x = self.Mixed_6c(x)
		# N x 768 x 17 x 17
		x = self.Mixed_6d(x)
		# N x 768 x 17 x 17
		x = self.Mixed_6e(x)
		# N x 768 x 17 x 17
		x = self.Mixed_7a(x)
		# N x 1280 x 8 x 8
		x = self.Mixed_7b(x)
		# N x 2048 x 8 x 8
		x = self.Mixed_7c(x)
		# N x 2048 x 8 x 8
		# Adaptive average pooling
		x = F.adaptive_avg_pool2d(x, (1, 1))
		# N x 2048 x 1 x 1
		x = F.dropout(x, training=self.training)
		# N x 2048 x 1 x 1
		x = torch.flatten(x, 1)
		# N x 2048
		x = self.fc(x)
		# N x 1 (num_classes)
		return x

	@torch.jit.unused
	def eager_outputs(self, x):
		return x

	def forward(self, x):
		x = self._transform_input(x)
		x = self._forward(x)
		if torch.jit.is_scripting():
			return InceptionOutputs(x)
		else:
			return self.eager_outputs(x)

	def classifier_parameters(self):
		return self.fc.parameters()

	def higher_level_parameters(self):
		hl_parameters = []
		hl_parameters += list(self.Mixed_7b.parameters())
		hl_parameters += list(self.Mixed_7a.parameters())
		return hl_parameters

	def lower_level_parameters(self):
		ll_parameters = []
		ll_parameters += list(self.Mixed_7c.parameters())
		ll_parameters += list(self.Mixed_6e.parameters())
		ll_parameters += list(self.Mixed_6d.parameters())
		ll_parameters += list(self.Mixed_6c.parameters())
		ll_parameters += list(self.Mixed_6b.parameters())
		ll_parameters += list(self.Mixed_6a.parameters())   
		ll_parameters += list(self.Mixed_5d.parameters())
		ll_parameters += list(self.Mixed_5c.parameters())
		ll_parameters += list(self.Mixed_5b.parameters())     
		return ll_parameters
	
	def unfreeze_classifier(self):
		for param in self.classifier_parameters():
			param.requires_grad = True

	def unfreeze_higher_level(self):
		for param in self.higher_level_parameters():
			param.requires_grad = True

	def unfreeze_lower_level(self):
		for param in self.lower_level_parameters():
			param.requires_grad = True


class Inception_A(nn.Module):

	def __init__(self, in_channels, pool_features, conv_block=None):
		super(Inception_A, self).__init__()
		if conv_block is None:
			conv_block = BasicConv2d
		self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

		self.branch3x3_1 = conv_block(in_channels, 48, kernel_size=1)
		self.branch3x3_2 = conv_block(48, 64, kernel_size=3, padding=1)

		self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
		self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
		self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

		self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

	def _forward(self, x):
		branch1x1 = self.branch1x1(x)

		branch3x3 = self.branch3x3_1(x)
		branch3x3 = self.branch3x3_2(branch3x3)

		branch3x3dbl = self.branch3x3dbl_1(x)
		branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
		branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

		branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
		branch_pool = self.branch_pool(branch_pool)

		outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
		return outputs

	def forward(self, x):
		outputs = self._forward(x)
		return torch.cat(outputs, 1)


class Inception_Reduction_A(nn.Module):

	def __init__(self, in_channels, conv_block=None):
		super(Inception_Reduction_A, self).__init__()
		if conv_block is None:
			conv_block = BasicConv2d
		self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

		self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
		self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
		self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

	def _forward(self, x):
		branch3x3 = self.branch3x3(x)

		branch3x3dbl = self.branch3x3dbl_1(x)
		branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
		branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

		branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

		outputs = [branch3x3, branch3x3dbl, branch_pool]
		return outputs

	def forward(self, x):
		outputs = self._forward(x)
		return torch.cat(outputs, 1)


class Inception_B(nn.Module):

	def __init__(self, in_channels, channels_7x7, conv_block=None):
		super(Inception_B, self).__init__()
		if conv_block is None:
			conv_block = BasicConv2d
		self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

		c7 = channels_7x7
		self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
		self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
		self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))

		self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
		self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
		self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
		self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
		self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))

		self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

	def _forward(self, x):
		branch1x1 = self.branch1x1(x)

		branch7x7 = self.branch7x7_1(x)
		branch7x7 = self.branch7x7_2(branch7x7)
		branch7x7 = self.branch7x7_3(branch7x7)

		branch7x7dbl = self.branch7x7dbl_1(x)
		branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
		branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
		branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
		branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

		branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
		branch_pool = self.branch_pool(branch_pool)

		outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
		return outputs

	def forward(self, x):
		outputs = self._forward(x)
		return torch.cat(outputs, 1)


class Inception_Reduction_B(nn.Module):

	def __init__(self, in_channels, conv_block=None):
		super(Inception_Reduction_B, self).__init__()
		if conv_block is None:
			conv_block = BasicConv2d
		self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
		self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

		self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
		self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
		self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
		self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

	def _forward(self, x):
		branch3x3 = self.branch3x3_1(x)
		branch3x3 = self.branch3x3_2(branch3x3)

		branch7x7x3 = self.branch7x7x3_1(x)
		branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
		branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
		branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

		branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
		outputs = [branch3x3, branch7x7x3, branch_pool]
		return outputs

	def forward(self, x):
		outputs = self._forward(x)
		return torch.cat(outputs, 1)


class Inception_C(nn.Module):

	def __init__(self, in_channels, conv_block=None):
		super(Inception_C, self).__init__()
		if conv_block is None:
			conv_block = BasicConv2d
		self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

		self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
		self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
		self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

		self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
		self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
		self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
		self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

		self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

	def _forward(self, x):
		branch1x1 = self.branch1x1(x)

		branch3x3 = self.branch3x3_1(x)
		branch3x3 = [
			self.branch3x3_2a(branch3x3),
			self.branch3x3_2b(branch3x3),
		]
		branch3x3 = torch.cat(branch3x3, 1)

		branch3x3dbl = self.branch3x3dbl_1(x)
		branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
		branch3x3dbl = [
			self.branch3x3dbl_3a(branch3x3dbl),
			self.branch3x3dbl_3b(branch3x3dbl),
		]
		branch3x3dbl = torch.cat(branch3x3dbl, 1)

		branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
		branch_pool = self.branch_pool(branch_pool)

		outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
		return outputs

	def forward(self, x):
		outputs = self._forward(x)
		return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

	def __init__(self, in_channels, out_channels, **kwargs):
		super(BasicConv2d, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
		self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return F.relu(x, inplace=True)
