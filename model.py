import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['Conv_Block', 'Conv_Block_Nested', 'UNetPlusPlus']


class Conv_Block(nn.Module): # this block is used for encoder network
	def __init__(self, in_channels, mid_channels, out_channels, pooling=False):
		super(Conv_Block, self).__init__()
		
		self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
		self.bn1 = nn.BatchNorm2d(mid_channels)
		self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pooling else None

	def forward(self, x):
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		if self.pool:
			out =  self.pool(out)
		return out


class Conv_Block_Nested(nn.Module): # this block is used for all other parts of network
	"""
	#
	#   Y -- O
	#       /  
	#      X
	#
	#
	"""
	def __init__(self, in_channels, mid_channels, out_channels, mode='nearest'):
		super(Conv_Block_Nested, self).__init__()

		self.mode = mode # upsampling method
		self.conv = Conv_Block(in_channels, mid_channels, out_channels)

	def forward(self, x, y):

		x = F.interpolate(x, scale_factor=2, mode=self.mode)
		if y:
			y = torch.cat(y, dim=1)
			x = torch.cat([x, y], dim=1)

		out = self.conv(x)
		return out


class UNetPlusPlus(nn.Module):
	def __init__(self, in_ch=3, out_ch=21, n_layers=5, mode='nearest'):
		super(UNetPlusPlus, self).__init__()

		self.mode = mode # upsampling method
		self.n_layers = n_layers

		n1 = 64
		filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

		# constructing the model
		self.model = nn.ModuleList()
		for i in range(self.n_layers): 
			tmp = nn.ModuleList()
			for j in range(self.n_layers-i):
				if j==0:
					if i==0:
						layer =  Conv_Block(in_ch, filters[0], filters[0], pooling=False)
					else:
						layer =  Conv_Block(filters[i-1], filters[i], filters[i], pooling=False)
				else: 
					layer = Conv_Block_Nested(filters[i]*j + filters[i+1], filters[i], filters[i])

				tmp.append(layer)
			self.model.append(tmp)

		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.last_conv = nn.Conv2d(filters[0], out_ch, kernel_size=1) 

	def forward(self, x):

		enc_features = list() # encoder features
		int_features = list() # intermediate features from previous layer

		# calculating encoder features
		for i in range(self.n_layers):
			x = self.model[i][0](x)
			if i!=0:
				x = self.pool(x)
			enc_features.append(x)

		int_features = [enc_features[-1]] # assign last year output as intermediate feature for layer n-1

		# calculating intermediate features row by row from bottom to top
		for i in range(self.n_layers-1, -1, -1):
			out = [enc_features[i]] # out is features from current layer [i]
			for j in range(1, self.n_layers-i):
				x = int_features[j-1]
				y = out[:j]
				out.append(self.model[i][j](x,y))

			int_features = out

		seg = self.last_conv(int_features[-1])
		return seg, int_features



if __name__ == '__main__':

	model = UNetPlusPlus()
	x = torch.rand(1, 3, 512, 512)
	result = model(x)