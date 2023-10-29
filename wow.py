import os
from re import I
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.image as mpimg
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix




model = Sequential()
def train_image():
    data_dir = '/Users/wzhang/Downloads/Data4'
    gasdata = pd.read_csv("/Users/wzhang/Downloads/data4.csv")
    labels = gasdata['Gas']
    imagedata = sorted(os.listdir(data_dir))
    X_data = []
    index = 0
    print(labels)
    for image in imagedata:
        img = mpimg.imread('/Users/wzhang/Downloads/Data4/'+image)
        img = img.reshape(320,320,9)
        img = img/255.0
        X_data.append(img)
    images = np.array(X_data)
 
    features_train, features_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2)
    features_train, features_validation, labels_train, labels_validation = train_test_split(features_train, labels_train, test_size=0.2)

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
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001), loss=tf.losses.BinaryCrossentropy(),metrics = ['accuracy'])
    print(model.summary())

    hist = model.fit(features_train, labels_train, epochs=10, validation_data=(features_validation, labels_validation))
    #hist.history

    fig = plt.figure()
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
    print(model.evaluate(features_test,labels_test))
    performance = model.predict(features_test)
    performance.round()
    actual = []
    for value in performance: 
        if(value>=0.5):
            actual.append(1)
        else:
            actual.append(0)
    print(actual)
    print(labels_test)
    cm = confusion_matrix(labels_test, actual)
 
    # sns.heatmap(cm, 
    #         annot=True,
    #         fmt='g', 
    #         xticklabels=['Gas','No Gas'],
    #         yticklabels=['Gas','No Gas'])
    # plt.ylabel('Prediction',fontsize=13)
    # plt.xlabel('Actual',fontsize=13)
    # plt.title('Confusion Matrix',fontsize=17)
    # plt.show()
    precision = precision_score(labels_test, actual)
    print('Precision: %f' % precision)
    recall = recall_score(labels_test, actual)
    print('Recall: %f' % recall)
    f1 = f1_score(labels_test, actual)
    print('F1 score: %f' % f1)


def image_test():
    data_dir = '/Users/wzhang/Downloads/No Gas'
    labels_dir = '/Users/wzhang/Downloads/ImageData'
    imagedata = sorted(os.listdir(sorted(data_dir)))
    labels = pd.read_csv(labels_dir)
    X_data = []
    for image in imagedata:
        img = mpimg.imread('/Users/wzhang/Downloads/No Gas/'+image)
        img = img.reshape(320,320,9)
        img = img/255.0
        X_data.append(img)
    images = np.array(X_data)
    print(model.evaluate(images,labels))
    performance = model.predict(images)
    performance.round()
    actual = []
    for value in performance: 
        if(value>=0.5):
            actual.append(1)
        else:
            actual.append(0)
    precision = precision_score(labels, actual)
    print('Precision: %f' % precision)
    recall = recall_score(labels, actual)
    print('Recall: %f' % recall)
    f1 = f1_score(labels, actual)
    print('F1 score: %f' % f1)

gasmodel = keras.Sequential([keras.layers.Dense(128, activation = 'relu', input_shape=(2,)),
                          keras.layers.Dense(20, activation=tf.nn.relu),
                         keras.layers.Dense(1,activation='sigmoid')])

def train_gas():
    data = pd.read_csv("/Users/wzhang/Downloads/data4.csv")
    labels = data['Gas']
    features = data.drop(columns=['Gas'])
    features_norm = (features - np.min(features)) / (np.max(features) - np.min(features))
    features_train, features_test, labels_train, labels_test = train_test_split(features_norm, labels, test_size=0.2)
    features_train, features_validation, labels_train, labels_validation = train_test_split(features_train, labels_train, test_size=0.2)
    gasmodel.compile(optimizer='adam',
             loss=tf.keras.losses.binary_crossentropy,
             metrics=['acc'])
    history = gasmodel.fit(features_train, labels_train, epochs=20, validation_data=(features_validation, labels_validation))

    history_dict = history.history
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
    plt.plot(history.history['acc'], color = 'red', label = 'acc')
    plt.plot(history.history['val_acc'], color = 'blue', label = 'acc')
    fig2.suptitle('Accuracy', fontsize = 20)
    plt.legend(loc = "upper left")
    plt.show()
    print(gasmodel.evaluate(features_test,labels_test))
    performance = gasmodel.predict(features_test)
    performance.round()
    actual = []
    for value in performance: 
        if(value>=0.5):
            actual.append(1)
        else:
            actual.append(0)
    print(actual)
    print(labels_test)
    # cm = confusion_matrix(labels_test, actual)
 
    # sns.heatmap(cm, 
    #         annot=True,
    #         fmt='g', 
    #         xticklabels=['Gas','No Gas'],
    #         yticklabels=['Gas','No Gas'])
    plt.ylabel('Prediction',fontsize=13)
    plt.xlabel('Actual',fontsize=13)
    plt.title('Confusion Matrix',fontsize=17)
    plt.show()
    precision = precision_score(labels_test, actual)
    print('Precision: %f' % precision)
    recall = recall_score(labels_test, actual)
    print('Recall: %f' % recall)
    f1 = f1_score(labels_test, actual)
    print('F1 score: %f' % f1)

def predict_gas():
    data = pd.read_csv("/Users/wzhang/Downloads/Gas Sensors Data - Sheet1 (3).csv")
    labels = data['Gas']
    labels = np.array(labels)
    features = data.drop(columns=['Gas'])
    print(labels)
    print(features)
    features_norm = (features - np.min(features)) / (np.max(features) - np.min(features))
    print(gasmodel.evaluate(features_norm,labels))
    performance = gasmodel.predict(features_norm)
    performance.round()
    actual = []
    for value in performance: 
        if(value>=0.5):
            actual.append(1)
        else:
            actual.append(0)
    print(actual)
    precision = precision_score(labels,actual)
    print('Precision: %f' % precision)
    recall = recall_score(labels,actual)
    print('Recall: %f' % recall)
    f1 = f1_score(labels, actual)
    print('F1 score: %f' % f1)
    for value in performance: 
        if(value>=0.5):
            print("Leak Detected")
        else:
            print("No Leak")
    print(actual)


finalmodel = Model()
def fusion(cnn, ann):
    x = Dense(4, activation="relu")(combinedInput)
    x = Dense(1, activation="linear")(x)
    finalmodel = Model(inputs=[cnn.input, ann.input], outputs=x)  
    opt = Adam(learning_rate=1e-3)
    finalmodel.compile(loss="mean_absolute_percentage_error", optimizer=opt)      


train_image()
predict_image()


