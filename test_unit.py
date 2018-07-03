from trainCNN import *
import os

if __name__ == '__main__':
	
	"""
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
	"""
	
	# Tune alpha
	os.system('python trainCNN.py -e 10 -a 1.3 -l2 1.25 -d 0.10 -bs 32 -sb1 3 -sb2 3')
	os.system('python trainCNN.py -e 10 -a 1.5 -l2 1.25 -d 0.10 -bs 32 -sb1 3 -sb2 3')
	os.system('python trainCNN.py -e 10 -a 2.0 -l2 1.25 -d 0.10 -bs 32 -sb1 3 -sb2 3')
	
	# Tune lr2
	os.system('python trainCNN.py -e 10 -a 1.5 -l2 1.5 -d 0.10 -bs 32 -sb1 3 -sb2 3')
	os.system('python trainCNN.py -e 10 -a 1.5 -l2 1.75 -d 0.10 -bs 32 -sb1 3 -sb2 3')
	os.system('python trainCNN.py -e 10 -a 1.5 -l2 2.0 -d 0.10 -bs 32 -sb1 3 -sb2 3')
	
	# Tune droput
	os.system('python trainCNN.py -e 10 -a 1.5 -l2 1.5 -d 0.05 -bs 32 -sb1 3 -sb2 3')
	os.system('python trainCNN.py -e 10 -a 1.5 -l2 1.5 -d 0.10 -bs 32 -sb1 3 -sb2 3')
	os.system('python trainCNN.py -e 10 -a 1.5 -l2 1.5 -d 0.15 -bs 32 -sb1 3 -sb2 3')
	os.system('python trainCNN.py -e 10 -a 1.5 -l2 1.5 -d 0.20 -bs 32 -sb1 3 -sb2 3')
	
	# Tune sizeblock1
	os.system('python trainCNN.py -e 10 -a 1.5 -l2 1.5 -d 0.15 -bs 32 -sb1 2 -sb2 3')
	os.system('python trainCNN.py -e 10 -a 1.5 -l2 1.5 -d 0.15 -bs 32 -sb1 3 -sb2 3')
	os.system('python trainCNN.py -e 10 -a 1.5 -l2 1.5 -d 0.15 -bs 32 -sb1 4 -sb2 3')
	
	# Tune sizeblock2
	os.system('python trainCNN.py -e 10 -a 1.5 -l2 1.5 -d 0.15 -bs 32 -sb1 3 -sb2 2')
	os.system('python trainCNN.py -e 10 -a 1.5 -l2 1.5 -d 0.15 -bs 32 -sb1 3 -sb2 4')
