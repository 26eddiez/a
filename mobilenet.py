from pyexpat import model
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
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, ReLU, DepthwiseConv2D, AvgPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
import locale
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from IPython.display import display
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from deepforest import CascadeForestClassifier
from sklearn.ensemble import RandomForestClassifier

gasdata = pd.read_csv("/Users/Eddie/Downloads/data5.csv")
data_dir = '/Users/Eddie/Downloads/fulldataset'
imagedata = sorted(os.listdir(data_dir))
print(len(imagedata))
X_data = []
for image in imagedata:
        # print(image)
        img = mpimg.imread('/Users/Eddie/Downloads/fulldataset/'+image)
        img.resize(224,224,3)
        img = img/255.0
        X_data.append(img)
images = np.array(X_data)
print(images.shape)
split = train_test_split(gasdata, images, test_size=0.2)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split
trainy = trainAttrX["Gas"]
testy = testAttrX["Gas"]
trainAttrX = trainAttrX.drop(columns=['Gas'])
testAttrX = testAttrX.drop(columns = ['Gas'])
trainAttrX= (trainAttrX - np.min(trainAttrX)) / (np.max(trainAttrX) - np.min(trainAttrX))
testAttrX = (testAttrX - np.min(testAttrX)) / (np.max(testAttrX) - np.min(testAttrX))
print(trainAttrX.shape)
print(testAttrX.shape)
print(trainy.shape)
print(testy.shape)
print(trainImagesX.shape)
print(trainy.shape)
# MobileNet block
def mobilnet_block (x, filters, strides):
    
    x = DepthwiseConv2D(kernel_size = 3, strides = strides, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv2D(filters = filters, kernel_size = 1, strides = 1)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

input = Input(shape = (224,224,3))
x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same')(input)
x = BatchNormalization()(x)
x = ReLU()(x)

# main part of the model
x = mobilnet_block(x, filters = 64, strides = 1)
x = mobilnet_block(x, filters = 128, strides = 2)
x = mobilnet_block(x, filters = 128, strides = 1)
x = mobilnet_block(x, filters = 256, strides = 2)
x = mobilnet_block(x, filters = 256, strides = 1)
x = mobilnet_block(x, filters = 512, strides = 2)
for _ in range (5):
    x = mobilnet_block(x, filters = 512, strides = 1)
x = mobilnet_block(x, filters = 1024, strides = 2)
x = mobilnet_block(x, filters = 1024, strides = 1)
x = AvgPool2D (pool_size = 7, strides = 1, data_format='channels_first')(x)
x = tf.keras.layers.Dropout(0.4)(x)
output= Flatten()(x)
output = Dense (4096, activation = "relu",kernel_regularizer='l2')(output)
output = (Dropout(0.4))(output)
output = Dense (2048, activation = "relu",kernel_regularizer='l2')(output)
output = Dense (1, activation = "sigmoid")(output)
model = Model(inputs=input, outputs=output)
# model.fit()
#mobilnet = create_cnn_mobilenet()
#mobilenet.compile(loss=None)
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), loss=tf.losses.BinaryCrossentropy(),metrics = ['acc']) 
print(model.input)
print(model.output)
print(trainy.shape)
# mobilenetPredict = mobilenet.predict(trainImagesX)
# print(mobilenetPredict.shape)
hist = model.fit(x=trainImagesX, y=trainy,validation_data=(testImagesX, testy),epochs=36)
model.save('/Users/Eddie/Downloads/alexnet')
performance = model.predict(testImagesX)
performance.round()
actual = []
for value in performance: 
    # print(value)
    if(value>=0.5):
            actual.append(1)
    else:
            actual.append(0)

acutal = np.array(actual)
# print(actual)
# print(testy)
accuracy = accuracy_score(testy, actual)
print('Accuracy: %f' % accuracy)
precision = precision_score(testy, actual)
print('Precision: %f' % precision)
recall = recall_score(testy, actual)
print('Recall: %f' % recall)
f1 = f1_score(testy, actual)
print('F1 score: %f' % f1)

history_dict = hist.history
history_dict.keys()
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'red', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

fig2 = plt.figure()
plt.plot(hist.history['acc'], color = 'red', label = 'acc')
plt.plot(hist.history['val_acc'], color = 'blue', label = 'acc')
fig2.suptitle('Accuracy', fontsize = 20)
plt.legend(loc = "upper left")
plt.show()