# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
def print_green(message):
	print(Fore.GREEN + str(message) + Style.RESET_ALL)

# Argument Parser
def parse_arguments():
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required=True,
		help="path to input dataset (i.e., directory of images)")
	ap.add_argument("-m", "--model", required=True,
		help="path to output model")
	ap.add_argument("-l", "--labelbin", required=True,
		help="path to output label binarizer")
	ap.add_argument("-p", "--plot", type=str, default="plot.png",
		help="path to output accuracy/loss plot")
	args = ap.parse_args()
	
	return args
	
def process_dataset(dataset_path, IMAGE_DIMS):
	data = []
	labels = []
	
	# Randomly shuffle the image paths
	print("[INFO] loading images...")
	image_paths = sorted(list(paths.list_images(dataset_path)))
	random.seed(42)
	random.shuffle(image_paths)
	
	if(IMAGE_DIMS[2] == 1):
		imread_flag = cv2.IMREAD_GRAYSCALE
	elif(IMAGE_DIMS[2] == 3):
		imread_flag = cv2.IMREAD_COLOR
	else:
		print_red('What are you trying to do? Only images with 1 or 2 channels allowed')
	
	# Load images into memmory
	for image_path in image_paths:
		# Read image, resize it to fit the CNN input size and convert to array
		image = cv2.imread(image_path, imread_flag)
		image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
		image = img_to_array(image)
		data.append(image)

		# Extrat the image label from the path and update label list
		label = image_path.split(os.path.sep)[-2]
		labels.append(label)
		
	# Normalize pixel data and convert to numpy arrays
	data = np.array(data, dtype="float") / 255.0
	labels = np.array(labels)
	print("[INFO] data size: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))
	
	# Create output matrix
	lb = LabelBinarizer()
	labels = lb.fit_transform(labels)
	
	# Partition dataset into training and testing (80%, 20%)
	trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)
	
	# Construct image generation for augmenting the dataset
	aug = ImageDataGenerator(
		rotation_range=25,
		width_shift_range=0.1,
		height_shift_range=0.1,
		shear_range=0.2,
		zoom_range=0.2,
		horizontal_flip=True,
		fill_mode="nearest"
	)
	
	return trainX, testX, trainY, testY, aug, lb

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
	trainX, testX, trainY, testY, aug, lb = process_dataset(args.dataset, IMAGE_DIMS)
	
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
