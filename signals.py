import numpy as np
from sklearn.preprocessing import scale
from scipy.interpolate import interp1d

'''
作者：费德里科·泰尔齐
该库包含处理信号所需的类。
这是一项正在进行的工作。。。。
'''


class Sample:

	"""
	传感器数据的加载和预处理

	采样用于加载、存储和处理从加速度计上获得的信号。
	它提供了一种从文件中加载信号并对其进行处理的方法。
	"""
	def __init__(self, acx, acy, acz, gx, gy, gz):
		self.acx = acx
		self.acy = acy
		self.acz = acz
		self.gx = gx
		self.gy = gy
		self.gz = gz

	def get_linearized(self, reshape = False):
		"""
		将数据线性化，组合6个不同的轴。用于将数据输入机器学习算法。
		如果restrape=True，它将对其进行重塑（在将其输入预测方法时非常有用）
		"""
		if reshape:
			return np.concatenate((self.acx, self.acy, self.acz, self.gx, self.gy, self.gz)).reshape(1, -1)
		else:
			return np.concatenate((self.acx, self.acy, self.acz, self.gx, self.gy, self.gz))

	@staticmethod
	def load_from_file(filename, size_fit=50):
		"""
		从文件加载信号数据。
		文件名：指示文件的路径。
		size_fit：是axe的最终样本数。
		它使用线性插值来增加或减少样本数量。
		"""
		# 将文件中的信号数据作为列表加载
		# 跳过第一行和最后一行，并将每个数字转换为整数
		data_raw = [list(map(lambda x: int(x), i.split(" ")[1:-1])) for i in open(filename)]
		# 将数据转换为浮点数
		data = np.array(data_raw).astype(float)
		# 通过缩放数据来标准化数据
		data_norm = scale(data)

		# 将每个axe提取到单独的变量中,这些表示3个轴上的加速度
		acx = data_norm[:, 0]
		acy = data_norm[:, 1]
		acz = data_norm[:, 2]

		# 这些表示在3个轴上的角度
		gx = data_norm[:, 3]
		gy = data_norm[:, 4]
		gz = data_norm[:, 5]

		# 为插值采样的每个轴创建一个函数
		x = np.linspace(0, data.shape[0], data.shape[0])
		f_acx = interp1d(x, acx)
		f_acy = interp1d(x, acy)
		f_acz = interp1d(x, acz)

		f_gx = interp1d(x, gx)
		f_gy = interp1d(x, gy)
		f_gz = interp1d(x, gz)

		# 通过对原版重新缩放创建具有所需样本大小的新样本集
		xnew = np.linspace(0, data.shape[0], size_fit)
		acx_stretch = f_acx(xnew)
		acy_stretch = f_acy(xnew)
		acz_stretch = f_acz(xnew)

		gx_stretch = f_gx(xnew)
		gy_stretch = f_gy(xnew)
		gz_stretch = f_gz(xnew)

		# 返回带有计算值的样本
		return Sample(acx_stretch, acy_stretch, acz_stretch, gx_stretch, gy_stretch, gz_stretch)
