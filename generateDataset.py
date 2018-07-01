from colorama import Fore, Style

import numpy as np
import cv2
import os

def print_green(message):
	print(Fore.GREEN + str(message) + Style.RESET_ALL)
	
def print_red(message):
	print(Fore.RED + str(message) + Style.RESET_ALL)
	
def process_line(line, emotion_map):
	line = line.split(',')
	return emotion_map[line[0]], np.array(line[1].split(), dtype = np.uint8), line[2]

if __name__ == '__main__':
	
	IMG_WIDTH = 48
	IMG_HEIGHT = 48
	
	emotion_map = {
		'0' : 'anger',
		'1' : 'disgust',
		'2' : 'fear',
		'3' : 'happiness',
		'4' : 'sadness',
		'5' : 'surprise',
		'6' : 'neutral'
	}
	
	img_count = {}
	for emotion in emotion_map.values():
		img_count[emotion] = 0
	
	f = open('fer2013/fer2013.csv')
	# Skip leading info line
	f.readline()
	
	# Create dataset dir if doesnt exists
	if(not os.path.exists('dataset')):
		os.mkdir('dataset')
	
	usages = {}
	for line in f:
		# Process line
		emotion, BWshade, usage = process_line(line, emotion_map)
		if usage not in usages:
			usages[usage] = 1
		else:
			usages[usage] += 1
		
		# Turn pixel data into image matrix
		image = BWshade.reshape( (IMG_HEIGHT, IMG_WIDTH) )
		
		# Create dir for this emotion if it doesnt exist
		if(not os.path.exists('dataset/{}'.format(emotion))):
			os.mkdir('dataset/{}'.format(emotion))
		
		# Store image
		cv2.imwrite('dataset/{}/{}{:05d}.png'.format(emotion, emotion, img_count[emotion]), image)
		
		# Update img_count
		img_count[emotion] += 1


		
