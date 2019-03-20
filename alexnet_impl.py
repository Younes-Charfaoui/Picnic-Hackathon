# importing the required libraries

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

class AlexNet():

	def __init__(self, image_size, num_classes):
		self.img_size = image_size
		self.num_classes = num_classes
		self.alexnet = Sequential()

		# Layer 1
		self.alexnet.add(Conv2D(96, (11, 11), input_shape=(self.img_size, self.img_size, 3), 
			padding='same', kernel_regularizer=l2(l2_reg)))
		self.alexnet.add(BatchNormalization())
		self.alexnet.add(Activation('relu'))
		self.alexnet.add(MaxPooling2D(pool_size=(2, 2)))

		# Layer 2
		self.alexnet.add(Conv2D(256, (5, 5), padding='same'))
		self.alexnet.add(BatchNormalization())
		self.alexnet.add(Activation('relu'))
		self.alexnet.add(MaxPooling2D(pool_size=(2, 2)))

		# Layer 3
		self.alexnet.add(ZeroPadding2D((1, 1)))
		self.alexnet.add(Conv2D(512, (3, 3), padding='same'))
		self.alexnet.add(BatchNormalization())
		self.alexnet.add(Activation('relu'))
		self.alexnet.add(MaxPooling2D(pool_size=(2, 2))

		# Layer 4
		self.alexnet.add(ZeroPadding2D((1, 1)))
		self.alexnet.add(Conv2D(1024, (3, 3), padding='same'))
		self.alexnet.add(BatchNormalization())
		self.alexnet.add(Activation('relu'))

		# Layer 5
		self.alexnet.add(ZeroPadding2D((1, 1)))
		self.alexnet.add(Conv2D(1024, (3, 3), padding='same'))
		self.alexnet.add(BatchNormalization())
		self.alexnet.add(Activation('relu'))
		self.alexnet.add(MaxPooling2D(pool_size=(2, 2)))

		# Layer 6
		self.alexnet.add(Flatten())
		self.alexnet.add(Dense(4096))
		self.alexnet.add(BatchNormalization())
		self.alexnet.add(Activation('relu'))
		self.alexnet.add(Dropout(0.4))

		# Layer 7
		self.alexnet.add(Dense(4096))
		self.alexnet.add(BatchNormalization())
		self.alexnet.add(Activation('relu'))
		self.alexnet.add(Dropout(0.4))

		# Layer 8
		self.alexnet.add(Dense(1024))
		self.alexnet.add(BatchNormalization())
		self.alexnet.add(Activation('relu'))
		self.alexnet.add(Dropout(0.4))

		# Layer 9
		self.alexnet.add(Dense(self.num_classes))
		self.alexnet.add(BatchNormalization())
		self.alexnet.add(Activation('softmax'))

	def get_model(self):
		return self.alexnet

		
