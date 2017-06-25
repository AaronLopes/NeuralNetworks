import numpy as np
from scipy import optimize

#Training Data
x_train = np.array([[3, 5], [5, 3], [10, 2]], dtype=float)
y_train = np.array([[75], [85], [93]], dtype=float)

#Testing Data
x_test = np.array([[4, 5.5], [4.5, 1], [9, 2.5], [6,2]], dtype=float)
y_test = np.array([[70], [89], [85], [75]], dtype=float)

#Normalize
x_train = x_train/np.amax(x_train, axis=0)
y_train = y_train/100

x_test = x_test/np.amax(x_train, axis=0)
y_test = y_test/100


class NeuralNetwork(object):
	def __init__(self, Lambda=0):
		# Define Hyperparameters
		self.inputLayerSize = 2
		self.outputLayerSize = 1
		self.hiddenLayerSize = 3

		# Weights (parameters)
		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

		# Regularization Parameter:
		self.Lambda = Lambda

	def forward(self, X):
		# Propogate inputs though network
		self.z2 = np.dot(X, self.W1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2, self.W2)
		yHat = self.sigmoid(self.z3)
		return yHat

	def sigmoid(self, z):
		# Apply sigmoid activation function to scalar, vector, or matrix
		return 1 / (1 + np.exp(-z))

	def sigmoidPrime(self, z):
		# Gradient of sigmoid
		return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

	def costFunction(self, X, y):
		# Compute cost for given X,y, use weights already stored in class.
		self.yHat = self.forward(X)
		J = 0.5 * sum((y - self.yHat) ** 2) / X.shape[0] + (self.Lambda / 2) * (
		np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
		return J

	def costFunctionPrime(self, X, y):
		# Compute derivative with respect to W and W2 for a given X and y:
		self.yHat = self.forward(X)

		delta3 = np.multiply(-(y - self.yHat), self.sigmoidPrime(self.z3))
		# Add gradient of regularization term:
		dJdW2 = np.dot(self.a2.T, delta3) / X.shape[0] + self.Lambda * self.W2

		delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
		# Add gradient of regularization term:
		dJdW1 = np.dot(X.T, delta2) / X.shape[0] + self.Lambda * self.W1

		return dJdW1, dJdW2

	# Helper functions for interacting with other methods/classes
	def getParams(self):
		# Get W1 and W2 Rolled into vector:
		params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		return params

	def setParams(self, params):
		# Set W1 and W2 using single parameter vector:
		W1_start = 0
		W1_end = self.hiddenLayerSize * self.inputLayerSize
		self.W1 = np.reshape(params[W1_start:W1_end], \
		                     (self.inputLayerSize, self.hiddenLayerSize))
		W2_end = W1_end + self.hiddenLayerSize * self.outputLayerSize
		self.W2 = np.reshape(params[W1_end:W2_end], \
		                     (self.hiddenLayerSize, self.outputLayerSize))

	def computeGradients(self, X, y):
		dJdW1, dJdW2 = self.costFunctionPrime(X, y)
		return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

class trainer(object):
	def __init__(self, N):
		self.N = N

	def costFunctionWrapper(self, params, x, y):
		self.N.setParams(params)
		cost = self.N.costFunction(x, y)
		grad = self.N.computeGradients(x, y)
		return cost, grad

	def callbackF(self, params):
		self.N.setParams(params)
		self.J.append(self.N.costFunction(self.x, self.y))

	def train(self, x, y):

		self.x = x
		self.y = y
		#List for costs
		self.J = []

		params0 = self.N.getParams()

		options = {'maxiter': 200, 'disp' : True}
		_res = optimize.minimize(self.costFunctionWrapper, params0,
		                         jac = True, method='BFGS', args=(x, y),
		                         options=options, callback=self.callbackF)
		self.N.setParams(_res.x)
		self.optimizeResults = _res

#For debugging gradient
def computeNumericalGradient(N, X, y):
	paramsInitial = N.getParams()
	numgrad = np.zeros(paramsInitial.shape)
	perturb = np.zeros(paramsInitial.shape)
	e = 1e-4

	for p in range(len(paramsInitial)):
		perturb[p] = e
		N.setParams(paramsInitial + perturb)
		loss2 = N.costFunction(X, y)

		N.setParams(paramsInitial - perturb)
		loss1 = N.costFunction(X, y)

		#Compute Numerical Gradient
		numgrad[p] = (loss2-loss1) / (2*e)

		perturb[p] = 0

	N.setParams(paramsInitial)
	return numgrad

NN = NeuralNetwork()

numgrad = computeNumericalGradient(NN, x_train, y_train)
grad = NN.computeGradients(x_train, y_train)
print(numgrad)
print(grad)

T = trainer(NN)
T.train(x_train, y_train)

print(NN.forward(x_test))
print(y_test)

