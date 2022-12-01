"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_classification

# X1为样本特征，Y1为样本类别输出， 共400个样本，每个样本2个特征，输出有3个类别，没有冗余特征，每个类别一个簇
X1, Y1 = make_classification(n_samples=400, n_features=2, n_redundant=0,
                             n_clusters_per_class=1, n_classes=3)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
plt.show()
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets._samples_generator import make_classification


# 生成数据
X, y = make_classification(n_samples=400, n_features=3, n_classes=3, n_redundant=0, n_informative=2, n_clusters_per_class=1, class_sep=2.0)

# 画样本的分布图像
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
ax.scatter(X[:,0], X[:,1], X[:,2], marker='o', c=y)
plt.show()
