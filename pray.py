from pyexpat import model
from IPython.display import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os
import locale
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
import locale
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

def create_cnn(width, height, depth, filters=(16, 32, 64)):
    model = Sequential()
    model.add(Conv2D(64,kernel_size=(3,3), activation = 'relu', input_shape = (320,320,9), kernel_regularizer = 'l2'))
    model.add(MaxPooling2D(pool_size = 2, strides = 2))
    model.add(Conv2D(32,kernel_size=(3,3),activation = 'relu'))
    model.add(MaxPooling2D(pool_size = 2, strides = 2))
    model.add(Conv2D(16,kernel_size=(3,3),activation = 'relu'))
    model.add(MaxPooling2D(pool_size = 2, strides = 2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(8,activation = 'relu'))
    model.add(Dense(4,activation = 'relu'))
    model.add(Dense(1,activation = 'sigmoid'))
    return model

def create_mlp(dim, regress=False):
	# define our MLP network
  model = Sequential()
  model.add(Dense(128, activation = 'relu', input_shape=(dim,),kernel_regularizer = 'l2'))
  model.add(Dense(64, activation = 'relu'))
  model.add(Dense(16, activation = tf.nn.relu))
  model.add(Dense(1, activation = 'softmax'))
  return model

gasdata = pd.read_csv("/Users/wzhang/Documents/data4.csv")
data_dir = '/Users/wzhang/Downloads/Data4'
imagedata = sorted(os.listdir(data_dir))
print(len(imagedata))
X_data = []
for image in imagedata:
    img = mpimg.imread('/Users/wzhang/Downloads/Data4/'+image)
    img = np.random.randint(0,255, (320,320,9),  dtype=np.uint32)
    img.reshape(320,320,9)
    normalized_image = img/255
    X_data.append(normalized_image)
images = np.array(X_data)
gasdata_norm = (gasdata - np.min(gasdata)) / (np.max(gasdata) - np.min(gasdata))

split = train_test_split(gasdata_norm, images, test_size=0.2)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split
trainy = trainAttrX["Gas"]
testy = testAttrX["Gas"]
trainAttrX = trainAttrX.drop(columns=['Gas'])
testAttrX = testAttrX.drop(columns = ['Gas'])

mlp = create_mlp(trainAttrX.shape[1], regress=False)
cnn = create_cnn(320, 320, 9)
combinedInput = concatenate([mlp.output, cnn.output])
print(mlp.output)
print(cnn.output)

x = Dense(32, activation="relu")(combinedInput)
x = Dense(2, activation = "sigmoid")(x)

model = Model(inputs=[mlp.input, cnn.input],outputs = x)
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), loss=tf.losses.BinaryCrossentropy(), metrics = ['accuracy']) 
hist = model.fit(x=[trainAttrX, trainImagesX], y=trainy,validation_data=([testAttrX, testImagesX], testy),epochs=10)
print(model.summary)
fig = plt.figure()

print(hist.history)

plt.plot(hist.history['loss'], color = 'teal', label = 'loss')
plt.plot(hist.history['val_loss'], color = 'orange', label = 'val_loss')
fig.suptitle('Loss', fontsize = 20)
plt.legend(loc = "upper left")
plt.show()

fig2 = plt.figure()
plt.plot(hist.history['accuracy'], color = 'teal', label = 'accuracy')
plt.plot(hist.history['val_accuracy'], color = 'orange', label = 'accuracy')
fig2.suptitle('Accuracy', fontsize = 20)
plt.legend(loc = "upper left")
plt.show()
accuracy = model.evaluate([testAttrX, testImagesX])
print(accuracy)

    



