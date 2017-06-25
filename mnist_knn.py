import numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_rows = x_train.reshape(x_train.shape[0], 28 * 28 )
x_test_rows = x_test.reshape(x_test.shape[0], 28 * 28 )

print('Number of training ')

class Nearest_Neighbor():

	def __init__(self):
		pass

	def train(self, x, y):
		self.x_train = x
		self.y_train = y

	def predict(self, x):
		num_test = x.shape[0]
		y_pred = np.zeros(num_test, dtype= self.y_train.dtype)

		for i in xrange(num_test):
			print("Test #:", i)
			distances = np.linalg.norm(self.x_train - x[i, :])
			min_index = np.argmin(distances)
			y_pred[i] = self.y_train[min_index]
			if i == 50:
				break
		return y_pred

NN = Nearest_Neighbor()
NN.train(x_train_rows, y_train)
y_test_predict = NN.predict(x_test_rows)
print 'accuracy %f' % (np.mean(y_test == y_test_predict) * 100)