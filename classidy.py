


def parse_arguments():
	ap = argparse.ArgumentParser()
	ap.add_argument("-m", "--model", required=True,
		help="path to trained model model")
	ap.add_argument("-l", "--labelbin", required=True,
		help="path to label binarizer")
	"""	
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	"""
	ap.add_argument("-d", "--data", required=True,
		help="path to folder with data to classify")
	args = ap.parse_args()
	
	return args
	
def process_image(img_path):
	image = cv2.imread(img_path)
	output = image.copy()
	
	# Process the image for classification
	image = cv2.resize(image, (96, 96))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	
	return image, output
	
def load_model_and_labels(model_path, labels_path):
	print("[INFO] loading network...")
	model = load_model(model_path)
	lb = pickle.loads(open(labels_path, "rb").read())
	
	return model, lb
	

if __name__ == '__main__':
	import argparse
	args = parse_arguments()
	
	# import the necessary packages
	from keras.preprocessing.image import img_to_array
	from keras.models import load_model
	from imutils import paths
	import numpy as np
	import imutils
	import pickle
	import cv2
	import os
	
	model, lb = load_model_and_labels(args.model, args.labelbin)
	
	img_paths = list(paths.list_images(args.data))
	
	print("[INFO] classifying images...\n")
	for img_path in img_paths:
		# Read Image
		image, output = process_image(img_path)
		img_name = img_path.split('/')[-1]
	
		# Classify the image
		proba = model.predict(image)[0]
		idx = np.argmax(proba)
		label = lb.classes_[idx]
	
		print('{} is {} with {:.2f}% probability\n'.format(img_name, label, proba[idx]*100))


