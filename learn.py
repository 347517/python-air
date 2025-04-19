"""
实现原理与逻辑总结：

1.数据预处理:通过读取文件，加载每个传感器数据样本，并进行数据标准化和插值，使得不同大小的样本能够统一处理
2.训练 SVM 模型：使用标准化和处理后的数据训练支持向量机（SVM）模型，进行分类任务。通过网格搜索 (GridSearchCV) 寻找最佳的超参数（C 和 kernel）。
3.评估模型性能：使用混淆矩阵和准确度对模型进行评估，帮助判断模型的分类效果。
4.模型保存：训练完成后，将模型和类别标签保存，以便后续使用，减少重复训练的开销。
"""

import sys

from signals import Sample
from sklearn import datasets	 # 用于加载机器学习数据集
from sklearn import svm		# 支持向量机分类器，用于创建模型
from sklearn.model_selection import train_test_split  # 用于将数据集分割成训练集和测试集
from sklearn.metrics import confusion_matrix  # 用于计算模型预测的混淆矩阵，评估分类性能
import scipy as sp
import os  # 用于文件和目录操作
import joblib  # 用于序列化（保存）和反序列化（加载）模型
from sklearn.model_selection import GridSearchCV  # 用于交叉验证和超参数调优

import numpy as np  # 数值计算库，广泛用于处理数据
from sklearn.preprocessing import scale  # 用于标准化数据，确保每个特征的均值为 0，方差为 1
from scipy.interpolate import interp1d  # 用于对数据进行插值，调整数据长度

"""
class Sample:

	def __init__(self, acx, acy, acz, gx, gy, gz):
		self.acx = acx
		self.acy = acy
		self.acz = acz
		self.gx = gx
		self.gy = gy
		self.gz = gz

	# 该方法将6个不同的轴数据（3个加速度计轴和3个陀螺仪轴）按顺序拼接成一个长向量，以便于输入到机器学习模型中
	def get_linearized(self, reshape=False):
		if reshape:
			return np.concatenate((self.acx, self.acy, self.acz, self.gx, self.gy, self.gz)).reshape(1, -1)
		else:
			return np.concatenate((self.acx, self.acy, self.acz, self.gx, self.gy, self.gz))

	# 该方法从指定的文件中加载原始的加速度计和陀螺仪数据。
	@staticmethod
	def load_from_file(filename, size_fit=50):
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
"""

SHOW_CONFUSION_MATRIX = False  # 该方法从指定的文件中加载原始的加速度计和陀螺仪数据。

# 分别存储输入特征和标签数据
x_data = []
y_data = []

# 存储类别和对应的标签
classes = {}

# 指定数据集存储根目录为data文件夹
root = "data"
print("从根目录加载数据集文件...".format(directory=root)),
# 从数据集的根目录获取所有数据文件
for path, subdirs, files in os.walk(root):
	for name in files:
		# 获取文件名
		filename = os.path.join(path, name)
		# 从文件中加载数据集文件，并处理成一个 Sample 对象
		sample = Sample.load_from_file(filename)
		# 使用 get_linearized() 方法将每个样本数据转换为适合机器学习模型的格式，并将数据集文件加载到x_data中
		x_data.append(sample.get_linearized())

		# 从文件名中提取类别，例如，文件“a_sample_0.txt”将被视为“a”
		category = name.split("_")[0]
		# 获取类别的编号，作为与类别的偏移量，到Ascii中的字符
		number = ord(category) - ord("a")
		# 将类别添加到y_data中
		y_data.append(number)
		# 将类别和相应的编号包含到词典中便于访问和参考
		classes[number] = category
print("数据集加载完成。")

# 交叉验证培训过程中使用的参数，库自动尝试每种可能的组合以，找到得分最好的。
params = {'C': [0.001, 0.01, 0.1, 1], 'kernel': ['linear']}

# 初始化分类器模型，probability = True为采用概率估计
svc = svm.SVC(probability=True)

# 使用？个处理核心和最大详细度对GridSearchCV进行本地化，GridSearchCV为网格搜索
clf = GridSearchCV(svc, params, verbose=10, n_jobs=1)
# 将数据集分成训练集，和测试集。35%的数据用作测试集，65% 用作训练集。random_state=0 确保每次分割的结果相同
X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.35, random_state=0)

print("开始训练过程...")

# 开始训练过程，用训练数据拟合分类器模型
clf.fit(X_train, Y_train)

"""
如果 SHOW_CONFUSION_MATRIX 为 True，则使用测试集数据进行预测，计算混淆矩阵。
混淆矩阵用于评估分类器的性能，显示预测结果与真实标签的对比。
"""
# 如果“显示混淆矩阵”为true，则print（）将显示混淆矩阵
if SHOW_CONFUSION_MATRIX:
	print("混淆矩阵:")
	Y_predicted = clf.predict(X_test)  # 对现有数据对其预测
	print(confusion_matrix(Y_test, Y_predicted))

print("最佳预测参数: ")
print(clf.best_estimator_)

# 计算找到的最佳估计器的分数。
score = clf.score(X_test, Y_test)
print("预测得分: {score}\n".format(score=score))

print("保存模型...")
joblib.dump(clf, 'model.pkl')  # 保存训练好的 SVM 模型
joblib.dump(classes, 'classes.pkl')  # 将类型保存到“classes.pkl”文件中
print("模型保存完成。")