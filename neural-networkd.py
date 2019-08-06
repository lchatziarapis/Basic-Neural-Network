from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import _pickle as cPickle
import pickle, sys, tarfile

#   Method for loading citaf-10 python version
#   In order for the function to work is to have
#   cifar-10-python.tar.gz in the same folder with the program.
#
def load(cv = 0, one_hot = True):

    f1 = tarfile.open("cifardata/cifar-10-python.tar.gz",'r:gz')
    training_set=[]
    for i in range(1,6):
        training_set.append(cPickle.load(f1.extractfile('cifar-10-batches-py/data_batch_%d' % (i)),encoding='latin1'))
    test_set = cPickle.load(f1.extractfile('cifar-10-batches-py/test_batch'),encoding='latin1')

    def to_one_hot(y):
        Y = np.zeros((len(y), 10))
        for i in range(len(y)):
            k = y[i]
            Y[i][k]=1
        return Y

    tot_trainX = [np.transpose(data['data'].reshape(-1,3,32,32),[0,2,3,1])
                        for data in training_set]
    if one_hot:
        tot_trainY = [to_one_hot(data['labels']) for data in training_set]
    else:
        tot_trainY = [data['labels'] for data in training_set]
    testX = np.transpose(
            test_set['data'].reshape(-1,3,32,32),[0,2,3,1])
    if one_hot:
        testY = to_one_hot(test_set['labels'])
    else:
        testY = test_set['labels']


    concate = False
    for i in range(5):
        if i==cv:
            validX = tot_trainX[i]
            validY = tot_trainY[i]
        elif concate:
            trainX = np.concatenate((trainX,
                tot_trainX[i]))
            trainY = np.concatenate((trainY,
                tot_trainY[i]))
        else:
            trainX = tot_trainX[i]
            trainY = tot_trainY[i]
            concate = True
    if cv==-1: return trainX, trainY, testX, testY
    return trainX, trainY, validX, validY, testX, testY

#   Loading the mnist files.
#   Returning four variables
#   train_data,train_truth: The training set
#   train_truth,test_truth:  The testing set
def load_MNIST():
    train_files = ['mnistdata/train%d.txt' % (i,) for i in range(10)]
    test_files = ['mnistdata/test%d.txt' % (i,) for i in range(10)]
    tmp = []
    for i in train_files:
    	with open(i,'r') as fp:
    		tmp += fp.readlines()
    train_data = np.array([[j for j in i.split(" ")] for i in tmp], dtype='int')/255

    print ("Train data array size: ", train_data.shape)

    tmp = []
    for i in test_files:
    	with open(i, 'r') as fp:
    		tmp += fp.readlines()
    # load test data in N*D array (10000x784 for MNIST)
    #                             divided by 255 to achieve normalization
    test_data = np.array([[j for j in i.split(" ")] for i in tmp], dtype='int') / 255
    print ("Test data array size: ", test_data.shape)

    tmp = []
    for i, _file in enumerate(train_files):
    	with open(_file, 'r') as fp:
    		for line in fp:
    			tmp.append([1 if j == i else 0 for j in range(0, 10)])
    train_truth = np.array(tmp, dtype='int')

    del tmp[:]
    for i, _file in enumerate(test_files):
    	with open(_file, 'r') as fp:
    		for _ in fp:
    			tmp.append([1 if j == i else 0 for j in range(0, 10)])
    test_truth = np.array(tmp, dtype='int')

    print ("Train truth array size: ", train_truth.shape)
    print ("Test truth array size: ", test_truth.shape)

    return train_data, test_data, train_truth, test_truth

#   Returning the chosen activation function and their derivatives for our model.
#   This function contains three basic functions:
#   1. log(1 + e^x)
#   2. (e^x - e^(-x)) / (e^x + e^(-x))
#   3. cos(x)
def activation_function(x,option):
    if(option == 1):
        return np.log(1 + np.exp(x)) , 1/(1 + np.exp(-x))
    elif(option == 2):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x)) , (4 * np.exp(2*x))/ ((np.exp(2*x) + 1)**2)
    elif(option == 3):
        return np.cos(x) , -1*np.sin(x)

#   Using the softmax for normalization
#   into a propabilitic distribution
#   using the y = exp(Wk.T x Xn) / sum(exp(Wj.T x Xn))
#   Different version of the softmax function for numerical stability
#   Simply substracting the maximum value from every element of a vector in order
#   to avoid underflow or overflow situations.
def softmax(y):
	max_of_rows = np.max(y, 1)
	m = np.array([max_of_rows, ] * y.shape[1]).T
	y = y - m
	y = np.exp(y)
	return y / (np.array([np.sum(y, 1), ] * y.shape[1])).T

#   Function that calculates the gradient for hidden and output layer.
#   Returns the two gradients and the cost function value.
#   For the calculation of the error function the logexpsum trick is used
def cost_gradient(X,t,W1,W2,lamda,option):
    #X  N x D+1
    #t  N x K
    #W1 M x D+1
    #W2 K x M+1
    # Forward propagation
    a1 = X.dot(W1.T) # N x M
    a1 = np.hstack((np.ones((a1.shape[0],1)), a1)) # N x M+1
    z1 = activation_function(a1,option)[0] # N x M+1
    a2 = z1.dot(W2.T) # N x K
    y = softmax(a2) # N x K

    dy = (t - y) # N x K

    delta2 = dy.dot(W2) * activation_function(a1,option)[1] # N x M+1
    delta2 = delta2[:,1:] # N x M

    gradW2 = dy.T.dot(z1) # K x M+1
    gradW1 = delta2.T.dot(X) # M x D+1

    gradW2 -= W2*lamda  # K x M+1
    gradW1 -= W1*lamda  # M x D+1

    max_error = np.max(y,axis=1)
    Ew = np.sum(t*a2) - np.sum(max_error) - \
        np.sum(np.log(np.sum(np.exp(a2-np.array([max_error,]*a2.shape[1]).T),1))) - \
        (0.5 * lamda) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

    return Ew,gradW1,gradW2

#   Main function for training the neural network
#   It seperates random batches from the original data
#   and perform training for a given iteration and error
def trainNetwork(X,t,W1,W2,lamda,options):
    iter = options[0]
    tol = options[1]
    eta = options[2]
    activ = options[3]
    batchN = options[4]
    Eword = -np.inf
    costs = []
    #   Loop for batching and training
    for i in range(iter):
        #   Random choice of 100 values
        batch = np.random.choice(X.shape[0],batchN)
        x_sample = X[batch]
        t_sample = t[batch]
        Ew,gradW1,gradW2 = cost_gradient(x_sample,t_sample,W1,W2,lamda,activ)
        W1 = W1 + eta * gradW1
        W2 = W2 + eta * gradW2

        costs.append(Ew)
        print('Iteration : %d, Cost function :%f' % (i, Ew))

        if np.abs(Ew - Eword) < tol:
        	break

        Eword = Ew

    return W1,W2,costs

#   Function for checking the our gradient Method
#   Our goal here is the return value to be small as possible
def gradient_check(X,t,W1,W2,lamda,option):
	epsilon = 1e-6
	_list = np.random.randint(X.shape[0], size=5)
	x_sample = np.array(X[_list, :])
	t_sample = np.array(t[_list, :])
	Ew,gradW1,gradW2 = cost_gradient(x_sample,t_sample,W1,W2,lamda,option)
	grad = np.concatenate((gradW1.ravel(),gradW2.ravel()))

	numericalGrad1 = np.zeros(gradW1.shape)
	for k in range(numericalGrad1.shape[0]):
		for d in range(numericalGrad1.shape[1]):
			w_tmp = np.copy(W1)
			w_tmp[k,d] += epsilon
			e_plus, _,_ = cost_gradient(x_sample,t_sample,w_tmp,W2,lamda,option)

			w_tmp = np.copy(W1)
			w_tmp[k,d] -= epsilon
			e_minus, _,_ = cost_gradient(x_sample,t_sample,w_tmp,W2,lamda,option)
			numericalGrad1[k, d] = (e_plus - e_minus) / (2 * epsilon)

	numericalGrad2 = np.zeros(gradW2.shape)
	for k in range(numericalGrad2.shape[0]):
		for d in range(numericalGrad2.shape[1]):
			w_tmp = np.copy(W2)
			w_tmp[k,d] += epsilon
			e_plus, _,_  = cost_gradient(x_sample,t_sample,W1,w_tmp,lamda,option)

			w_tmp = np.copy(W2)
			w_tmp[k,d] -= epsilon
			e_minus, _,_  = cost_gradient(x_sample,t_sample,W1,w_tmp,lamda,option)
			numericalGrad2[k, d] = (e_plus - e_minus) / (2 * epsilon)

	num = np.concatenate((numericalGrad1.ravel(),numericalGrad2.ravel()))
	error = np.max(np.abs(grad - num))
	print ("The difference estimate for gradient of w is : ", error)

#   Getting error for calculating with Y_test
def error(X,W1,W2,option):
	# Forward propagation
	a1 = X.dot(W1.T) # 60000 x 100
	a1 = np.hstack((np.ones((a1.shape[0],1)), a1)) # 60000 x 101
	z1 = activation_function(a1,option)[0] # 60000 x 101
	a2 = z1.dot(W2.T) #  60000 x 10
	y = softmax(a2)

	error = np.argmax(y,1)
	return error

data_choice = int(input("Do you want MNIST-CIFAR(0 - 1): "))
print("Loading files...")
if(data_choice == 0):
    X_train, X_test, Y_train, Y_test = load_MNIST()
else:
	X_train, Y_train, X_test, Y_test = load(cv=-1,one_hot=True)
	X_train = X_train.reshape(X_train.shape[0], 32 * 32 * 3)/255
	X_test = X_test.reshape(X_test.shape[0], 32 * 32 * 3)/255


N,D = X_train.shape
X_train = np.hstack((np.ones((X_train.shape[0],1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0],1)), X_test))

hidden_size = int(input("Choose your hidden layer size: "))
M,K = hidden_size,10

# Instead of having weights zeroed I simply apply a random distribution
# for their initialization
W1 = 0.1*np.random.randn(M,D)
W1 = np.hstack((np.ones((W1.shape[0],1)), W1))

W2 = 0.1*np.random.randn(K,M)
W2 = np.hstack((np.ones((W2.shape[0],1)), W2))


initChoice  = int(input("Training data or checking gradient?(0 - 1): "))

if(initChoice == 0):
    print ("Activation Functions")
    print ("  "+ "1.log(1 + e^a)\n"+"  "+"2.(e^a - e^(-a)) / (e^a + e^(-a))\n"+"  "+"3.cos(a)")
    func_option = int(input("Choose your activation function: "))




    # Options:
    #   1 - Number of iterations
    #   2 - Number of tol error
    #   3 - Number of lamda
    #   4 - Number of function(Check activation function)
    #   5 - Number of batches(100 - 200)
    options = [5000, 1e-6,1/N,func_option,100]


    W1,W2,costs = trainNetwork(X_train,Y_train,W1,W2,0.1,options)

    ttest = error(X_test,W1,W2,func_option)
    error_count = np.not_equal(np.argmax(Y_test,1),ttest).sum()
    print ("Error is ", error_count / Y_test.shape[0] * 100, " %")

    plt.plot(np.abs(np.squeeze(costs)))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(format(options[2], 'f')))
    plt.show()
else:
    func_option = int(input("Choose your activation function: "))
    gradient_check(X_train,Y_train,W1,W2,0.1,func_option)
