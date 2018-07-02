# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
def print_green(message):
	print(Fore.GREEN + str(message) + Style.RESET_ALL)
	
def print_red(message):
	print(Fore.RED + str(message) + Style.RESET_ALL)

# Argument Parser
def parse_arguments():
	ap = argparse.ArgumentParser()
	ap.add_argument("-td", "--training_dataset", required=True,
		help="path to training dataset (i.e., directory of images)")
	ap.add_argument("-vd", "--validation_dataset", required=True,
		help="path to validation dataset (i.e., directory of images)")
	ap.add_argument("-m", "--model", required=True,
		help="path to output model")
	ap.add_argument("-l", "--labelbin", required=True,
		help="path to output label binarizer")
	ap.add_argument("-p", "--plot", type=str, default="plot.png",
		help="path to output accuracy/loss plot")
	args = ap.parse_args()
	
	return args
	
def get_shuffled_paths(training_dataset_path, validation_dataset_path):
	training_image_paths = sorted(list(paths.list_images(training_dataset_path)))
	validation_image_paths = sorted(list(paths.list_images(validation_dataset_path)))
	random.seed(42)
	random.shuffle(training_image_paths)
	random.shuffle(validation_image_paths)
	
	return training_image_paths, validation_image_paths
	
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
	
	
def process_dataset(training_dataset_path, validation_dataset_path, IMAGE_DIMS):
	training_inp = []
	validation_inp = []
	training_labels = []
	validation_labels = []
	
	print("[INFO] loading images...")
	# Randomly shuffle the image paths
	t_image_paths, v_image_paths = get_shuffled_paths(training_dataset_path, validation_dataset_path)
	
	# Set correct flag for colored or grayscale images
	imread_flag = get_imread_flag(IMAGE_DIMS)
	
	# Load training images into memory
	for t_image_path in t_image_paths:
		# Read image, resize it to fit the CNN input size and convert to array
		image = cv2.imread(t_image_path, imread_flag)
		image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
		image = img_to_array(image)
		training_inp.append(image)

		# Extrat the image label from the path and update label list
		label = t_image_path.split(os.path.sep)[-2]
		training_labels.append(label)
	training_inp = np.array(training_inp)
		
	# Load validation images into memory
	for v_image_path in v_image_paths:
		# Read image, resize it to fit the CNN input size and convert to array
		image = cv2.imread(v_image_path, imread_flag)
		image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
		image = img_to_array(image)
		validation_inp.append(image)

		# Extrat the image label from the path and update label list
		label = v_image_path.split(os.path.sep)[-2]
		validation_labels.append(label)
	validation_inp = np.array(validation_inp)
	
	# Get label lists as numpy arrays
	training_labels = np.array(training_labels)
	validation_labels = np.array(validation_labels)
	
	# Printo info about training and validation data
	print("[INFO] training data size: {:.2f}MB".format(training_inp.nbytes / (1024 * 1000.0)))
	print("[INFO] validation data size: {:.2f}MB".format(validation_inp.nbytes / (1024 * 1000.0)))
	
	# Create output matrices
	lb = LabelBinarizer()
	training_out = lb.fit_transform(training_labels)
	validation_out = lb.fit_transform(validation_labels)
	
	# Construct object for data augmentation
	aug = ImageDataGenerator(
		# z-score Normalization
		featurewise_center = True,
		featurewise_std_normalization = True,
		
		# Image transformation
		horizontal_flip = True,
		vertical_flip = True,
		width_shift_range = 0.2,
		height_shift_range = 0.2,
	)
	aug.fit(training_inp)

	return training_inp, validation_inp, training_out, validation_out, aug, lb

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

def get_normalized(data):
	mean = np.mean(data, axis=0, dtype='float')
	stddev = np.std(data, axis=0, dtype='float')
	data = data - mean
	data = data / stddev

	return data
	
if __name__ == '__main__':
	import argparse	
	# Get arguments
	args = parse_arguments()
	
	# import the necessary packages
	from colorama import Fore, Style
	from keras.preprocessing.image import ImageDataGenerator
	from keras.optimizers import Adam
	from keras.preprocessing.image import img_to_array
	from sklearn.preprocessing import LabelBinarizer
	from sklearn.model_selection import train_test_split
	import matplotlib.pyplot as plt
	from imutils import paths
	import numpy as np
	import random
	import pickle
	import cv2
	import os
	# Import model
	from CNN.buildModel import CNN
	
	# Initialize constants
	EPOCHS = 10
	INIT_ALPHA = 0.001
	L2_RATE = 0.01
	N = 3
	M = 4
	DROPOUT_RATE = 0.25
	BATCH_SIZE = 32	# LOOK FOR THIS
	IMAGE_DIMS = (48, 48, 1)
	
	# Process images for training
	trainX, testX, trainY, testY, aug, lb = process_dataset(args.training_dataset, args.validation_dataset, IMAGE_DIMS)
	
	# Normalize validation data with z-score
	testX = (testX - aug.mean)/aug.std
	
	# Initialize the model
	print("[INFO] compiling model...")
	model = CNN.buildDeeperCNN(
		width=IMAGE_DIMS[1],
		height=IMAGE_DIMS[0],
		depth=IMAGE_DIMS[2],
		classes=len(lb.classes_),
		n = N,
		m = M,
		l2rate = L2_RATE,
		dropout_rate = DROPOUT_RATE
	)
	# Optimizer
	opt = Adam(lr=INIT_ALPHA, decay=INIT_ALPHA / EPOCHS)
	
	# Compile the model
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
	
	# Train the network
	print("[INFO] training network...")
	H = model.fit_generator(
		aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
		validation_data=(testX, testY),
		steps_per_epoch=len(trainX) // BATCH_SIZE,
		epochs=EPOCHS,
		verbose=1
	)
	
	# Save the model
	print("[INFO] saving model...")
	model.save(args.model)
	 
	# Save the label binarizer
	print("[INFO] saving label binarizer...")
	f = open(args.labelbin, "wb")
	f.write(pickle.dumps(lb))
	f.close()
	
	# Generate trining loss and accuracy plot
	save_loss_and_accuracy_plot(EPOCHS, H, args.plot)
