#=========================================================================#
#	Modules
#=========================================================================#
# Import utilites
from utils import *

# Get arguments before importing the rest of packages so that it will
# fail swiftly in case of argument error
args = parse_train_arguments()

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import Adam
import numpy as np
import cv2
import os

# Import model
from CNN.buildModel import CNN

#=========================================================================#
#	Functions
#=========================================================================#
def process_dataset(training_dataset_path, validation_dataset_path, IMAGE_DIMS):
	training_inp = []
	validation_inp = []
	training_labels = []
	validation_labels = []
	
	printy("[INFO] loading images...")
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
	printy("[INFO] training data size: {:.2f}MB".format(training_inp.nbytes / (1024 * 1000.0)))
	printy("[INFO] validation data size: {:.2f}MB".format(validation_inp.nbytes / (1024 * 1000.0)))
	
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

def trainCNN(args, IMAGE_DIMS = (48, 48, 1)):
	# Process images for training
	if(args.short_test):
		trainX, testX, trainY, testY, aug, lb = process_dataset('dataset/PublicTest', 'dataset/PublicTest', IMAGE_DIMS)
	else:
		trainX, testX, trainY, testY, aug, lb = process_dataset('dataset/Training', 'dataset/PublicTest', IMAGE_DIMS)
	
	# Normalize validation data with z-score
	testX = (testX - aug.mean)/aug.std
	
	# Initialize the model
	printy("[INFO] compiling model...")
	model = CNN.buildDeeperCNN(
		width=IMAGE_DIMS[1],
		height=IMAGE_DIMS[0],
		depth=IMAGE_DIMS[2],
		classes=len(lb.classes_),
		n = args.sizeblock1,
		m = args.sizeblock2,
		l2rate = args.l2regrate,
		dropout_rate = args.dropout
	)
	# Optimizer
	opt = Adam(lr=args.alpha, decay=args.alpha / args.epochs)
	
	# Compile the model
	model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
	
	# Train the network
	printy("[INFO] training network...")
	H = model.fit_generator(
		aug.flow(trainX, trainY, batch_size=args.batchsize),
		validation_data=(testX, testY),
		steps_per_epoch=len(trainX) // args.batchsize,
		epochs=args.epochs,
		verbose=1
	)
	
	# Save obtained model with it's labels and loss/acurracy plot
	printy("[INFO] saving obtained model and results...")
	save_results_to_disk(model, lb, H, args)
	printg('Model succesfully trained!')
	
#=========================================================================#
#	Main
#=========================================================================#
	
if __name__ == '__main__':
	trainCNN(args)
