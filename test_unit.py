import os

if __name__ == '__main__':
	
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
