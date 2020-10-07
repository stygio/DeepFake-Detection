import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):

	def __init__(self, in_channels, out_channels, **kwargs):
		super(BasicConv2d, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
		self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		return F.leaky_relu(x, negative_slope = 0.2, inplace = True)


class Reseption2(nn.Module):

	def __init__(self, num_classes=1, init_weights=True):
		super(Reseption2, self).__init__()

		self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2)
		self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
		self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
		self.MaxPool_3a = nn.MaxPool2d(3, stride=2)
		self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
		self.Conv2d_4a_3x3 = BasicConv2d(80, 160, kernel_size=3)
		self.Conv2d_4b_3x3 = BasicConv2d(160, 224, kernel_size=3, stride=2)
		# self.MaxPool_5a = nn.MaxPool2d(3, stride=2)
		self.Reseption_A_1 = Reseption_A(224, scale_factor = 1)
		self.Reseption_A_2 = Reseption_A(224, scale_factor = 1)
		self.Reseption_A_3 = Reseption_A(224, scale_factor = 1)
		self.Reduction_A = Reseption_Reduction_A(224)
		self.Reseption_B_1 = Reseption_B(784, scale_factor = 1)
		self.Reseption_B_2 = Reseption_B(784, scale_factor = 1)
		self.Reseption_B_3 = Reseption_B(784, scale_factor = 1)
		self.Reseption_B_4 = Reseption_B(784, scale_factor = 1)
		self.Reduction_B = Reseption_Reduction_B(784)
		self.Reseption_C_1 = Reseption_C(1568, scale_factor = 1)
		self.Reseption_C_2 = Reseption_C(1568, scale_factor = 1)
		self.fc = nn.Linear(1568, num_classes)

		if init_weights:
			for m in self.modules():
				if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
					nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('leaky_relu', 0.2))
				elif isinstance(m, nn.BatchNorm2d):
					nn.init.constant_(m.weight, 1)
					nn.init.constant_(m.bias, 0)

	def forward(self, x):
		# N x 3 x 299 x 299
		x = self.Conv2d_1a_3x3(x)
		# N x 32 x 149 x 149
		x = self.Conv2d_2a_3x3(x)
		# N x 32 x 147 x 147
		x = self.Conv2d_2b_3x3(x)
		# N x 64 x 147 x 147
		x = self.MaxPool_3a(x)
		# N x 64 x 73 x 73
		x = self.Conv2d_3b_1x1(x)
		# N x 80 x 73 x 73
		x = self.Conv2d_4a_3x3(x)
		# N x 160 x 71 x 71
		x = self.Conv2d_4b_3x3(x)
		# N x 224 x 35 x 35
		x = self.Reseption_A_1(x)
		# N x 224 x 35 x 35
		x = self.Reseption_A_2(x)
		# N x 224 x 35 x 35
		x = self.Reseption_A_3(x)
		# N x 224 x 35 x 35
		x = self.Reduction_A(x)
		# N x 784 x 17 x 17
		x = self.Reseption_B_1(x)
		# N x 784 x 17 x 17
		x = self.Reseption_B_2(x)
		# N x 784 x 17 x 17
		x = self.Reseption_B_3(x)
		# N x 784 x 17 x 17
		x = self.Reseption_B_4(x)
		# N x 784 x 17 x 17
		x = self.Reduction_B(x)
		# N x 1568 x 8 x 8
		x = self.Reseption_C_1(x)
		# N x 1568 x 8 x 8
		x = self.Reseption_C_2(x)
		# N x 1568 x 8 x 8
		# Adaptive average pooling
		x = F.adaptive_avg_pool2d(x, (1, 1))
		# N x 1568 x 1 x 1
		x = torch.flatten(x, 1)
		# N x 1568 x 1 x 1
		x = F.dropout(x, p = 0.2, training=self.training)
		# N x 1568
		x = self.fc(x)
		# N x num_classes
		return x

	def classifier_parameters(self):
		return self.fc.parameters()

	def higher_level_parameters(self):
		hl_parameters = []
		return hl_parameters

	def lower_level_parameters(self):
		ll_parameters = []    
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


class Reseption_A(nn.Module):

	def __init__(self, in_channels, scale_factor = 1.):
		super(Reseption_A, self).__init__()
		self.scale_factor = scale_factor

		branch1x1_out_channels = 32
		self.branch1x1 = BasicConv2d(in_channels, branch1x1_out_channels, kernel_size=1)

		branch3x3_out_channels = 32
		self.branch3x3_1 = BasicConv2d(in_channels, 32, kernel_size=1)
		self.branch3x3_2 = BasicConv2d(32, branch3x3_out_channels, kernel_size=3, padding=1)

		branch3x3dbl_out_channels = 48
		self.branch3x3dbl_1 = BasicConv2d(in_channels, 32, kernel_size=1)
		self.branch3x3dbl_2 = BasicConv2d(32, 32, kernel_size=3, padding=1)
		self.branch3x3dbl_3 = BasicConv2d(32, branch3x3dbl_out_channels, kernel_size=3, padding=1)

		self.combine_branches = BasicConv2d(branch1x1_out_channels + branch3x3_out_channels + branch3x3dbl_out_channels, 224, kernel_size=1)

	def forward(self, x):
		branch1x1 = self.branch1x1(x)

		branch3x3 = self.branch3x3_1(x)
		branch3x3 = self.branch3x3_2(branch3x3)

		branch3x3dbl = self.branch3x3dbl_1(x)
		branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
		branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

		y = torch.cat([branch1x1, branch3x3, branch3x3dbl], 1)
		y = self.combine_branches(y)
		y = y * self.scale_factor + x

		return F.leaky_relu(y, negative_slope = 0.2, inplace = True)


class Reseption_Reduction_A(nn.Module):

	def __init__(self, in_channels):
		super(Reseption_Reduction_A, self).__init__()

		self.branch3x3 = BasicConv2d(in_channels, 336, kernel_size=3, stride=2)

		self.branch3x3dbl_1 = BasicConv2d(in_channels, 128, kernel_size=1)
		self.branch3x3dbl_2 = BasicConv2d(128, 128, kernel_size=3, padding=1)
		self.branch3x3dbl_3 = BasicConv2d(128, 224, kernel_size=3, stride=2)

		self.max_pool = nn.MaxPool2d(3, stride=2)

	def forward(self, x):
		branch3x3 = self.branch3x3(x)

		branch3x3dbl = self.branch3x3dbl_1(x)
		branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
		branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

		branch_pool = self.max_pool(x)

		return torch.cat([branch3x3, branch3x3dbl, branch_pool], 1)


class Reseption_B(nn.Module):

	def __init__(self, in_channels, scale_factor = 1.):
		super(Reseption_B, self).__init__()
		self.scale_factor = scale_factor
		
		branch1x1_out_channels = 128
		self.branch1x1 = BasicConv2d(in_channels, branch1x1_out_channels, kernel_size=1)

		branch7x7_channels = 128
		self.branch7x7_1 = BasicConv2d(in_channels, branch7x7_channels, kernel_size=1)
		self.branch7x7_2 = BasicConv2d(branch7x7_channels, branch7x7_channels, kernel_size=(1, 7), padding=(0, 3))
		self.branch7x7_3 = BasicConv2d(branch7x7_channels, branch7x7_channels, kernel_size=(7, 1), padding=(3, 0))

		branch7x7dbl_channels = 128
		self.branch7x7dbl_1 = BasicConv2d(in_channels, branch7x7dbl_channels, kernel_size=1)
		self.branch7x7dbl_2 = BasicConv2d(branch7x7dbl_channels, branch7x7dbl_channels, kernel_size=(7, 1), padding=(3, 0))
		self.branch7x7dbl_3 = BasicConv2d(branch7x7dbl_channels, branch7x7dbl_channels, kernel_size=(1, 7), padding=(0, 3))
		self.branch7x7dbl_4 = BasicConv2d(branch7x7dbl_channels, branch7x7dbl_channels, kernel_size=(7, 1), padding=(3, 0))
		self.branch7x7dbl_5 = BasicConv2d(branch7x7dbl_channels, branch7x7dbl_channels, kernel_size=(1, 7), padding=(0, 3))

		self.combine_branches = BasicConv2d(branch1x1_out_channels + branch7x7_channels + branch7x7dbl_channels, 784, kernel_size=1)

	def forward(self, x):
		branch1x1 = self.branch1x1(x)

		branch7x7 = self.branch7x7_1(x)
		branch7x7 = self.branch7x7_2(branch7x7)
		branch7x7 = self.branch7x7_3(branch7x7)

		branch7x7dbl = self.branch7x7dbl_1(x)
		branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
		branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
		branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
		branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

		y = torch.cat([branch1x1, branch7x7, branch7x7dbl], 1)
		y = self.combine_branches(y)
		y = y * self.scale_factor + x

		return F.leaky_relu(y, negative_slope = 0.2, inplace = True)


class Reseption_Reduction_B(nn.Module):

	def __init__(self, in_channels):
		super(Reseption_Reduction_B, self).__init__()

		# self.branch3x3_1 = BasicConv2d(in_channels, 392, kernel_size=1)
		# self.branch3x3_2 = BasicConv2d(392, 392, kernel_size=3, stride=2)

		# self.branch7x7x3_1 = BasicConv2d(in_channels, 392, kernel_size=1)
		# self.branch7x7x3_2 = BasicConv2d(392, 392, kernel_size=(1, 7), padding=(0, 3))
		# self.branch7x7x3_3 = BasicConv2d(392, 392, kernel_size=(7, 1), padding=(3, 0))
		# self.branch7x7x3_4 = BasicConv2d(392, 392, kernel_size=3, stride=2)

		self.branch3x3_1 = BasicConv2d(in_channels, 256, kernel_size=1)
		self.branch3x3_2 = BasicConv2d(256, 392, kernel_size=3, stride=2)

		self.branch7x7x3_1 = BasicConv2d(in_channels, 256, kernel_size=1)
		self.branch7x7x3_2 = BasicConv2d(256, 256, kernel_size=(1, 7), padding=(0, 3))
		self.branch7x7x3_3 = BasicConv2d(256, 320, kernel_size=(7, 1), padding=(3, 0))
		self.branch7x7x3_4 = BasicConv2d(320, 392, kernel_size=3, stride=2)

		self.max_pool = nn.MaxPool2d(3, stride=2)

	def forward(self, x):
		branch3x3 = self.branch3x3_1(x)
		branch3x3 = self.branch3x3_2(branch3x3)

		branch7x7x3 = self.branch7x7x3_1(x)
		branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
		branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
		branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

		branch_pool = self.max_pool(x)

		return torch.cat([branch3x3, branch7x7x3, branch_pool], 1)


class Reseption_C(nn.Module):

	def __init__(self, in_channels, scale_factor=1.):
		super(Reseption_C, self).__init__()
		self.scale_factor = scale_factor

		branch1x1_out_channels = 192
		self.branch1x1 = BasicConv2d(in_channels, branch1x1_out_channels, kernel_size=1)

		branch3x3_out_channels = 128
		self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
		self.branch3x3_2a = BasicConv2d(192, branch3x3_out_channels, kernel_size=(1, 3), padding=(0, 1))
		self.branch3x3_2b = BasicConv2d(192, branch3x3_out_channels, kernel_size=(3, 1), padding=(1, 0))

		branch3x3dbl_out_channels = 128
		self.branch3x3dbl_1 = BasicConv2d(in_channels, 192, kernel_size=1)
		self.branch3x3dbl_2 = BasicConv2d(192, 192, kernel_size=3, padding=1)
		self.branch3x3dbl_3a = BasicConv2d(192, branch3x3dbl_out_channels, kernel_size=(1, 3), padding=(0, 1))
		self.branch3x3dbl_3b = BasicConv2d(192, branch3x3dbl_out_channels, kernel_size=(3, 1), padding=(1, 0))

		self.combine_branches = BasicConv2d(branch1x1_out_channels + 2*branch3x3_out_channels + 2*branch3x3dbl_out_channels, 1568, kernel_size=1)

	def forward(self, x):
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

		y = torch.cat([branch1x1, branch3x3, branch3x3dbl], 1)
		y = self.combine_branches(y)
		y = y * self.scale_factor + x

		return F.leaky_relu(y, negative_slope = 0.2, inplace = True)

