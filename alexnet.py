from pyexpat import model
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
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
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
def create_cnn():
    model = Sequential()
    model.add(Conv2D(filters = 96, input_shape = (224, 224, 3),  
                kernel_size = (11, 11), strides = (4, 4),  
                padding = 'valid',kernel_regularizer = 'l2')) 
    model.add(Activation('relu')) 
    # Max-Pooling  
    model.add(MaxPooling2D(pool_size = (2, 2), 
                strides = (2, 2), padding = 'valid')) 
    # Batch Normalisation 
    model.add(BatchNormalization()) 
    
    # 2nd Convolutional Layer 
    model.add(Conv2D(filters = 256, kernel_size = (11, 11),  
                strides = (1, 1), padding = 'valid')) 
    model.add(Activation('relu')) 
    # Max-Pooling 
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2),  
                padding = 'valid')) 
    # Batch Normalisation 
    model.add(BatchNormalization()) 
    
    # 3rd Convolutional Layer 
    model.add(Conv2D(filters = 384, kernel_size = (3, 3),  
                strides = (1, 1), padding = 'valid')) 
    model.add(Activation('relu')) 
    # Batch Normalisation 
    model.add(BatchNormalization()) 
    
    # 4th Convolutional Layer 
    model.add(Conv2D(filters = 384, kernel_size = (3, 3),  
                strides = (1, 1), padding = 'valid')) 
    model.add(Activation('relu')) 
    # Batch Normalisation 
    model.add(BatchNormalization()) 
    
    # 5th Convolutional Layer 
    model.add(Conv2D(filters = 256, kernel_size = (3, 3),  
                strides = (1, 1), padding = 'valid')) 
    model.add(Activation('relu')) 
    # Max-Pooling 
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2),  
                padding = 'valid')) 
    # Batch Normalisation 
    model.add(BatchNormalization()) 
    
    # Flattening 
    model.add(Flatten()) 
    
    # 1st Dense Layer 
    model.add(Dropout(0.4)) 
    model.add(Dense(4096,kernel_regularizer='l2')) 
    model.add(Activation('relu') )
    # Add Dropout to prevent overfitting 
    model.add(Dropout(0.4)) 
    # Batch Normalisation 
    model.add(BatchNormalization()) 
    
    # 2nd Dense Layer 
    model.add(Dense(2048,kernel_regularizer='l2') )
    model.add(Activation('relu')) 
    # Add Dropout 
    # Batch Normalisation 
    model.add(BatchNormalization()) 
    model.add(Dense(1,activation = 'sigmoid'))
    return model
gasdata = pd.read_csv("/Users/Eddie/Downloads/data5.csv")
data_dir = '/Users/Eddie/Downloads/fulldataset'
imagedata = sorted(os.listdir(data_dir))
print(len(imagedata))
X_data = []
for image in imagedata:
        #print(image)
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

# print(testy)
newtesty = []
for value in testy:
    value = float(value)
    newtesty.append(value)
# print(newtesty)
newtrainy = []
for value in trainy:
    value = float(value)
    newtrainy.append(value)

print(newtrainy)
model = create_cnn()
#model.load('/Users/Eddie/Downloads/alexnet')
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001), loss='binary_crossentropy',metrics = ['acc'])
hist = model.fit(trainImagesX,trainy,validation_data=(testImagesX,testy),epochs=60,batch_size = 20)

model.save('/Users/Eddie/Downloads/alexnet')
#model = tf.saved_model.load('/Users/Eddie/Downloads/alexnet')

performance = model.predict(testImagesX)
performance.round()
actual = []
for value in performance: 
    #print(value)
    if(value>=0.5):
            actual.append(1)
    else:
            actual.append(0)

acutal = np.array(actual)
print(testy)
accuracy = accuracy_score(testy,actual)
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
