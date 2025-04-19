import sys

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import scipy as sp
import os
import joblib
import signals
from sklearn.model_selection import GridSearchCV

SHOW_CONFUSION_MATRIX = False
x_data = []
y_data = []

classes = {}
# 数据集存储根目录data文件夹
root = "data"
print("从根目录加载数据集文件...".format(directory=root)),
# 从数据集的根目录获取所有数据文件
for path, subdirs, files in os.walk(root):
	for name in files:
		# 获取文件名
		filename = os.path.join(path, name)
		# 从文件中加载数据集文件
		sample = signals.Sample.load_from_file(filename)
		# 将数据集文件加载到x_data中
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

# 初始化模型，probability = True为采用概率估计
svc = svm.SVC(probability=True)
# 使用8个处理核心和最大详细度对GridSearchCV进行本地化，GridSearchCV为网格搜索
clf = GridSearchCV(svc, params, verbose=10, n_jobs=8)
# 将数据集分成两个子集，一个用于培训，一个用于测试（测试集占35%）
X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.35, random_state=0)

print("开始训练过程...")

# 开始训练过程，用训练数据拟合分类器模型
clf.fit(X_train, Y_train)

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
joblib.dump(clf, 'model.pkl')  # 将模型保存到“model.pkl”文件中
joblib.dump(classes, 'classes.pkl')  # 将类型保存到“classes.pkl”文件中
print("模型保存完成。")