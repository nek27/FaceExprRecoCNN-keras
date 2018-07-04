from trainCNN import *
import decimal

def drange(x, jump, y):
  while x <= y:
    yield float(x)
    #x += decimal.Decimal(jump)
    x += jump

epochs = 10

alpha_i = 1.2
alpha_e = 1.8

l2r_i = 1.2
l2r_e = 2

do_i = 0.05
do_e = 0.35

batchsize = 32

sizeb1_i = 2
sizeb1_e = 4

sizeb2_i = 2
sizeb2_e = 4

if __name__ == '__main__':
	trainX, testX, trainY, testY, aug, lb = load_dataset()
	
	"""
	# Test alpha
	for alpha in drange(alpha_i, 0.1, alpha_e):
		trainCNN(trainX, testX, trainY, testY, aug, lb, epochs = 1, alpha = alpha)
	"""
	
	# Test l2r
	for l2r in drange(l2r_i, 0.1, l2r_e):
		trainCNN(trainX, testX, trainY, testY, aug, lb, l2r = l2r)
		
	# Test dropout
	for dropout in drange(do_i, 0.05, do_e):
		trainCNN(trainX, testX, trainY, testY, aug, lb, dropout = dropout)
		
	# Test size of block 1
	for sizeb1 in drange(sizeb1_i, 1, sizeb1_e):
		trainCNN(trainX, testX, trainY, testY, aug, lb, sizeb1 = sizeb1)
		
	for sizeb2 in drange(sizeb2_i, 1, sizeb2_e):
		trainCNN(trainX, testX, trainY, testY, aug, lb, sizeb2 = sizeb2)
