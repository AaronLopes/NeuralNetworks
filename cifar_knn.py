import numpy as np
import matplotlib.pyplot as plt
from cs231n.data_utils import load_CIFAR10
from cs231n.classifiers import KNearestNeighbor

#Load data

cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_tre, y_tre, X_te, y_te = load_CIFAR10(cifar10_dir)
X_tre_rows = X_tre.reshape(X_tre.shape[0], 32*32*3)
X_te_rows = X_te.reshape(X_te.shape[0], 32*32*3)

X_val_rows = X_tre_rows[:1000, :]
y_val_rows = y_tre[:1000]

X_tre_rows = X_tre_rows[1000:, :]
y_tre = y_tre[:1000]

val_acc = []
for k in [1, 3, 5, 10, 20, 50, 100]:

	knn = KNearestNeighbor()
	knn.train(X_tre, y_tre)

	y_val_predict = knn.predict(X_val_rows, k=k)
	acc = np.mean(y_val_predict == y_val_rows)
	print 'accuracy: %f' % (acc*100)

	val_acc.append((k, acc))
