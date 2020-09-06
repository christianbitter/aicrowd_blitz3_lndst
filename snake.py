import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv
from datetime import datetime


def init():
	tf.get_logger().setLevel('ERROR')
	print("Tensorflow: ", tf.__version__)
	gpus = tf.config.experimental.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(gpus[0], True)

def build_gen():
	train_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255., rotation_range=60, brightness_range=(0.2, 0.8), vertical_flip=True, 
	width_shift_range=0.2, height_shift_range=0.2, dtype='float32')

	val_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255., dtype='float32')
	test_data_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255., dtype='float32')	

	return train_data_gen, val_data_gen, test_data_gen


def get_model(no_classes=2, w=224, h=224, c=3, lr=0.03):
	conv_filters = [32, 64, 128, 256]	

	m = keras.Sequential(name='AIBlit-Snake')
	m.add(keras.layers.Conv2D(input_shape=(w,h,c), kernel_size=(3,3), filters=conv_filters[0], name="Conv0"))
	m.add(keras.layers.BatchNormalization())
	m.add(keras.layers.Activation(activation=tf.nn.relu))
	m.add(keras.layers.MaxPool2D(pool_size=(2,2)))

	for i, f in enumerate(conv_filters[1:]):		
		m.add(keras.layers.Conv2D(kernel_size=(3,3), filters=f, name="Conv{}".format(i+1)))
		m.add(keras.layers.BatchNormalization())
		m.add(keras.layers.Activation(activation=tf.nn.relu))		
		m.add(keras.layers.MaxPool2D(pool_size=(2,2)))
		
	m.add(keras.layers.Flatten())
	m.add(keras.layers.Dense(units=1024, activation=tf.nn.relu))
	m.add(keras.layers.Dense(units=no_classes, activation=tf.nn.sigmoid))

	m.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss=keras.losses.binary_crossentropy, metrics=["acc"])

	return m

def write_csv(rows, file_path:str, header=["id", "class"]):
	with open(file_path, 'w', newline='') as csvfile:
		w = csv.writer(csvfile)
		w.writerow(header)		
		for r in rows:
			w.writerow(r)

def main():
	init()

	batch_size = 10
	learning_rate = 0.03
	no_epochs = 10

	w, h, c = 224, 224, 3	
	no_classes = 2

	plot_result = False
	use_small   = False

	# images are 224x224
	train_dir_fp = "C:/Development/data/AIBlitz/Snake{}/train".format("_small" if use_small else "")
	test_dir_fp  = "C:/Development/data/AIBlitz/Snake{}".format("_small" if use_small else "")
	val_dir_fp   = "C:/Development/data/AIBlitz/Snake{}/val".format("_small" if use_small else "")

	print("Data Configuration")
	print("Training Data: {}".format(train_dir_fp))
	print("Testing Data: {}".format(test_dir_fp))
	print("Validation Data: {}".format(val_dir_fp))

	img_target_shape = (w, h)
	train_data_gen, val_data_gen, test_data_gen = build_gen()

	model = get_model(no_classes=no_classes, w=w, h=h, c=c)
	model.summary()

	class_names = ["non_venomous", "venomous"]

	print("Start Training: {}".format(datetime.now()))

	fit_history = model.fit(train_data_gen.flow_from_directory(directory=train_dir_fp, target_size=img_target_shape, shuffle=True), epochs=no_epochs, 
	validation_data=val_data_gen.flow_from_directory(directory=val_dir_fp, target_size=img_target_shape, shuffle=True), 
	callbacks=[keras.callbacks.ReduceLROnPlateau(), keras.callbacks.ModelCheckpoint(filepath="AI_Blitz/Snake/ModelCheckPoint/model.{epoch:02d}-{val_loss:.2f}.hdf5", save_weights_only=False)])

	print("End Training: {}".format(datetime.now()))
		
	test_gen = test_data_gen.flow_from_directory(directory=test_dir_fp, target_size=img_target_shape, class_mode=None, shuffle=False, classes=['test'])
	# for i in test_gen:
		# idx = (test_gen.batch_index - 1) * test_gen.batch_size
		# print(test_gen.filenames[idx : idx + test_gen.batch_size])
	predictions = model.predict_generator(test_gen)		
	header      = ["id", "class"]
	rows        = []
	for i, (pred, fname) in enumerate(zip(predictions, test_gen.filenames)):
		# todo: remove the file extension
		imgid = os.path.basename(fname)
		cname = class_names[np.argmax(pred)]
		# print("Prediction ({}: {}) -> {}".format(i, fname, pred))
		row = [imgid, cname]
		rows.append(row)
		
	write_csv(rows, file_path="./test_submission.csv")

	if plot_result:
		history = fit_history.history
		plt.figure(1)
		plt.subplot(211)
		plt.plot(history["loss"], label='training')
		plt.plot(history["val_loss"], label='validation')
		plt.title("Learning Loss")
		plt.legend()
		plt.subplot(212)
		plt.plot(history["acc"], label='training')
		plt.plot(history["val_acc"], label='validation')	
		plt.title("Classifier Accuracy")
		plt.legend()
		plt.show()	

if __name__ == "__main__":
	main()