# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# Import required packages
import matplotlib.pyplot as plt
from colorama import Fore
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# Color print functions
def printg(message):
	print('{}{}{}'.format(Fore.GREEN, message, Fore.RESET))
	
def printr(message):
	print('{}{}{}'.format(Fore.RED, message, Fore.RESET))
	
def printy(message):
	print('{}{}{}'.format(Fore.YELLOW, message, Fore.RESET))
	
# trainCNN.py argument parser
def parse_train_arguments():
	desc = '\
Module with functions designed to train the DeeperCNN described in the\
Facial Expresion Recognition with Convolutional Neural Networks paper by\
Arushi Raghuvanshi and Vivek Choksi from the Standford University'
	ap = argparse.ArgumentParser(fromfile_prefix_chars = '@', description = desc)
	
	ap.add_argument("-e", "--epochs", required=False, type=int, default=10,
		help="int: amount of epochs to train for")
	ap.add_argument("-a", "--alpha", required=False, type=float, default=0.001,
		help="float: initial amount for alpha")
	ap.add_argument("-l2", "--l2regrate", required=False, type=float, default=0.01,
		help="float: L2 regularization rate (aka L2 alpha)")
	ap.add_argument("-d", "--dropout", required=False, type=float, default=0.25,
		help="float: fraction of synapses to dropout during training")
	ap.add_argument("-bs", "--batchsize", required=False, type=int, default=32,
		help="int: size of batches for every training iteration")
	ap.add_argument("-sb1", "--sizeblock1", required=False, type=int, default=3,
		help="int: amount of layers to use in the first block")
	ap.add_argument("-sb2", "--sizeblock2", required=False, type=int, default=4,
		help="int: amount of layers to use in the second block")
	ap.add_argument("-s", "--short_test", required=False, action='store_true',
		help="run short test using PublicTest as training and validation dataset")
	return ap.parse_args()

# Returns arrays with the images in training_dataset_[ath and
# validation_dataset_path randomly shuffled
def get_shuffled_paths(training_dataset_path, validation_dataset_path):
	training_image_paths = sorted(list(paths.list_images(training_dataset_path)))
	validation_image_paths = sorted(list(paths.list_images(validation_dataset_path)))
	random.seed(42)
	random.shuffle(training_image_paths)
	random.shuffle(validation_image_paths)
	
	return training_image_paths, validation_image_paths

# Returns appropiate cv2 imread flag for the images pixel dimension	
def get_imread_flag(IMAGE_DIMS):
	imread_flag = None
	if(IMAGE_DIMS[2] == 1):
		imread_flag = cv2.IMREAD_GRAYSCALE
	elif(IMAGE_DIMS[2] == 3):
		imread_flag = cv2.IMREAD_COLOR
	else:
		print_red('What are you trying to do? Only images with 1 or 3 channels allowed')
		exit()
		
	return imread_flag

# Saves plot with loss and accuracy from the last epoch of training
def save_loss_and_accuracy_plot(EPOCHS, H, plot_path):
	plt.style.use("ggplot")
	plt.figure()
	N = EPOCHS
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="upper left")
	plt.savefig(plot_path)

# Saves all the results in a folder under the folder it is executed in	
def save_results_to_disk(model, lb, H, args):
	folder_st = 'res_{}e_{}a_{}l2_{}d_{}bs_{}sb1_{}sb2'.format(
		args.epochs, args.alpha, args.l2regrate, args.dropout,
		args.batchsize, args.sizeblock1, args.sizeblock2)

	if(not os.path.exists(folder_st)):
		os.mkdir(folder_st)

	# Save the model
	model.save('{}/model.model'.format(folder_st))
	 
	# Save the label binarizer
	f = open('{}/labels.pickle'.format(folder_st), "wb")
	f.write(pickle.dumps(lb))
	f.close()
	
	# Generate training loss and accuracy plot
	save_loss_and_accuracy_plot(args.epochs, H, '{}/loss_acc_plot.png'.format(folder_st))
	
	# Store log on a txt
	buff = ''
	for i in range(args.epochs):
		buff += 'EPOCH {}\n'.format(i)
		for key in H.history.keys():
			buff += '{}: {} '.format(key, H.history[key][i])
		buff +='\n'
	f= open("{}/training_log.txt".format(folder_st), 'w')
	f.write(buff)
	f.close()
	
	# Store log in a .pickle for later use
	f = open('{}/training_log_dict.pickle'.format(folder_st), 'wb')
	pickle.dump(H.history, f)
	f.close()
		
		
