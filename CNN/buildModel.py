# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.regularizers import l2
from keras.initializers import he_normal
from keras import backend as K

class CNN(object):
	
	@staticmethod
	#def buildSmallerVGG(width, height, depth, classes):
	def build(width, height, depth, classes):
		
		# Initialize model
		model = Sequential()
		
		# Define inputShape and configure format to match keras'
		if(K.image_data_format() == 'channels_first'):
			input_shape = (depth, height, width)
			chan_idx = 1
		else:
			input_shape = (height, width, depth)
			chan_idx = -1
			
		# Build model structure
		# Step 1
		model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = input_shape))
		model.add(Activation('relu'))
		
		model.add(BatchNormalization(axis = chan_idx))
		model.add(MaxPooling2D(pool_size = (3, 3)))
		
		model.add(Dropout(0.25))
		
		# Step 2
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		
		model.add(BatchNormalization(axis = chan_idx))
		
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		
		model.add(BatchNormalization(axis = chan_idx))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		
		model.add(Dropout(0.25))
		
		# Step 3
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		
		model.add(BatchNormalization(axis = chan_idx))
		
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		
		model.add(BatchNormalization(axis = chan_idx))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		
		model.add(Dropout(0.25))
		
		# Classification Step
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation("relu"))
		
		model.add(BatchNormalization())
		
		model.add(Dropout(0.5))
		
		# Softmax clasifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		
		return model
		
		
		
		
	@staticmethod
	def buildDeeperCNN(width, height, depth, classes, n, m, l2rate, dropout_rate):
		# Initialize model
		model = Sequential()
		
		# Define inputShape and configure format to match keras'
		if(K.image_data_format() == 'channels_first'):
			input_shape = (depth, height, width)
			chan_idx = 1
		else:
			input_shape = (height, width, depth)
			chan_idx = -1
			
		# First block
		model.add(Conv2D(32, (3, 3), input_shape = input_shape,
						 padding = 'same', strides = 1, use_bias = True,
						 kernel_initializer = 'he_normal',
						 bias_initializer = 'he_normal',
						 kernel_regularizer = l2(l2rate),
						 bias_regularizer = l2(l2rate)))
		model.add(BatchNormalization(axis = chan_idx))
		model.add(Activation('relu'))
		
		# Second block
		for n in range(0, n):
			model.add(Conv2D(32, (3, 3), padding = 'same', strides = 1,
							 kernel_initializer='he_normal',
							 bias_initializer='he_normal',
							 kernel_regularizer = l2(l2rate),
							 bias_regularizer = l2(l2rate)))
			model.add(BatchNormalization(axis = chan_idx))
			model.add(Activation('relu'))
			
			model.add(Conv2D(32, (3, 3), padding = 'same', strides = 1,
							 kernel_initializer='he_normal',
							 bias_initializer='he_normal',
							 kernel_regularizer = l2(l2rate),
							 bias_regularizer = l2(l2rate)))
			model.add(BatchNormalization(axis = chan_idx))
			model.add(Activation('relu'))
			
			model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
			model.add(Dropout(dropout_rate))
			
		# Third block
		for n in range(0, m):
			model.add(Conv2D(64, (3, 3), padding = 'same', strides = 1,
							 kernel_initializer='he_normal',
							 bias_initializer='he_normal',
							 kernel_regularizer = l2(l2rate),
							 bias_regularizer = l2(l2rate)))
			model.add(BatchNormalization(axis = chan_idx))
			model.add(Activation('relu'))
			
			model.add(Conv2D(64, (3, 3), padding = 'same', strides = 1,
							 kernel_initializer='he_normal',
							 bias_initializer='he_normal',
							 kernel_regularizer = l2(l2rate),
							 bias_regularizer = l2(l2rate)))
			model.add(BatchNormalization(axis = chan_idx))
			model.add(Activation('relu'))
			
			#model.add(MaxPooling2D(pool_size = (2, 2), strides = 2))
			model.add(MaxPooling2D(pool_size = (2, 2), strides = 1))
			model.add(Dropout(dropout_rate))
			
		# Fourth block
		model.add(Flatten())
		model.add(Dense(512, kernel_initializer='he_normal',
						bias_initializer='he_normal',
						kernel_regularizer = l2(l2rate),
						bias_regularizer = l2(l2rate)))
		model.add(BatchNormalization(axis = chan_idx))
		model.add(Activation('relu'))
		model.add(Dropout(dropout_rate))
		
		model.add(Dense(256, kernel_initializer='he_normal',
						bias_initializer='he_normal',
						kernel_regularizer = l2(l2rate),
						bias_regularizer = l2(l2rate)))
		model.add(BatchNormalization(axis = chan_idx))
		model.add(Activation('relu'))
		model.add(Dropout(dropout_rate))
		
		model.add(Dense(128, kernel_initializer='he_normal',
						bias_initializer='he_normal',
						kernel_regularizer = l2(l2rate),
						bias_regularizer = l2(l2rate)))
		model.add(BatchNormalization(axis = chan_idx))
		model.add(Activation('relu'))
		model.add(Dropout(dropout_rate))
		
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		
		return model
