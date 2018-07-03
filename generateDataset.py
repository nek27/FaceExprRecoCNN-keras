import numpy as np
import cv2
import os
	
def process_line(line, emotion_map):
	line = line.split(',')
	return emotion_map[line[0]], np.array(line[1].split(), dtype = np.uint8), line[2][:-1]

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
	
	for line in f:
		# Process line
		emotion, BWshade, usage = process_line(line, emotion_map)
		
		# Turn pixel data into image matrix
		image = BWshade.reshape( (IMG_HEIGHT, IMG_WIDTH) )
		
		# Create dirs for the stablished structure if they dont exist
		if(not os.path.exists('dataset/{}'.format(usage))):
			os.mkdir('dataset/{}'.format(usage))
		
		if(not os.path.exists('dataset/{}/{}'.format(usage, emotion))):
			os.mkdir('dataset/{}/{}'.format(usage, emotion))
		
		# Store image
		cv2.imwrite('dataset/{}/{}/{}{:05d}.png'.format(usage, emotion, emotion, img_count[emotion]), image)
		
		# Update img_count
		img_count[emotion] += 1


		
