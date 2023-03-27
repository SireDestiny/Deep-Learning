# -*- coding: utf-8 -*-
"""
Created on Sun May 23 01:03:19 2021

@author: Dell
"""

import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from numpy.random import seed
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

## TASK 1----------------------------------------------------------------------
with np.load("train_data_label.npz") as data:
    train_data=data['train_data']
    train_label=data['train_label']
    
with np.load("test_data_label.npz") as data:
    test_data=data['test_data']
    test_label=data['test_label']
    

seed(1)
X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2, random_state=42)

X_train_knn=X_train/255
X_val_knn=X_val/255
test_data_knn=test_data/255
n_neighbors = [3,5,7,9,11,13,15,17,19]
# fit KNN on training set and try on validation set
pred_knn=[]
for k in n_neighbors:
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_knn, y_train)
    pred_knn+=[knn.predict(X_val_knn)]
    
#function to get error rate on validation set
def evaluate(gold, predicted):
    total_errors=0
    for i in range(len(gold)):
        if predicted[i]!= gold[i]:
            total_errors+=1
    error_rate= total_errors/len(gold)
    return error_rate 

#get the error rate of predictions on validation set
error_rates_all_k=[]
for pred_k in pred_knn:
    error_rates_all_k+=[evaluate(y_val, pred_k)]
    
#plot to show which k value produces the lowest validation error
x=n_neighbors
y=error_rates_all_k
plt.plot(x, y)
plt.xlabel("k values")
plt.ylabel("error rates of all k values")
plt.xlim(2)
plt.xticks(x)
plt.show()

# with k=3, the validation error is minimum, so k = 3 is the best hyperparameter, 
# so we took n_neighbors=3 for predicting on new data
seed(1)
knn2= KNeighborsClassifier(n_neighbors=3)
knn2.fit(X_train, y_train)
preds_knn=knn2.predict(test_data)
accuracy_knn=accuracy_score(test_label,preds_knn)

#plot confusion matrix for KNN performance

CM_knn= confusion_matrix(test_label, preds_knn)
plt.figure(figsize = (15,15))
sns.heatmap(CM_knn, annot=True, cmap="Blues", fmt = 'g', xticklabels=np.unique(test_label), yticklabels=np.unique(test_label))
plt.xlabel("Predicted Classes")
plt.ylabel("True Classes")
plt.title("Confusion Matrix")
plt.show()


def class_scores(y_true, y_pred, reference):
    """Function which takes two lists and a reference indicating which class 
    to calculate the TP, FP, and FN for."""
    Y_true = set([ i for (i,v) in enumerate(y_true) if v == reference])
    Y_pred = set([ i for (i,v) in enumerate(y_pred) if v == reference])
    TP = len(Y_true.intersection(Y_pred))
    FP = len(Y_pred - Y_true)
    FN = len(Y_true - Y_pred)
    return TP, FP, FN

# recall per class:
labels=[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
recall_knn=[]
for t in labels:
    TP, FP, FN=class_scores(test_label.tolist(), preds_knn.tolist(),t)
    recall_knn+=[TP/(TP+FN)]

    
recall_dict_knn={}
for l,r in zip(labels,recall_knn):
    recall_dict_knn[l]=r
print(recall_dict_knn)

# find the letters that are easiest to classify (letters having recall score=1)
indices_easy_knn = [i for i, x in enumerate(recall_knn) if x == 1]
easy_knn=[]
for ind in indices_easy_knn:
    easy_knn+=[labels[ind]]
easy_letters_knn=set(easy_knn)

# find the letter that is the most difficult to classify (Letter having lowest recall score)
indices_hard_knn = [i for i, x in enumerate(recall_knn) if x == min(recall_knn)]
hard_knn=[]
for ind in indices_hard_knn:
    hard_knn+=[labels[ind]]
hard_letters_knn=set(hard_knn)

        
size  = 28
channels = 1
batch = 128
epochs = 100

seed(1)
X_train, X_val, y_train, y_val = train_test_split(train_data, train_label, test_size=0.2, random_state=42)

X_train_cnn= X_train.reshape(X_train.shape[0], size, size, channels)

X_val_cnn=X_val.reshape(X_val.shape[0], size, size, channels)
test_data_cnn=test_data.reshape(test_data.shape[0],size,size,channels)

def build_model(in_shape):
    model = Sequential([Conv2D(filters=16,  kernel_size=(3,3), activation="relu", input_shape=in_shape),
                    MaxPooling2D(2,2, padding='same'),
                    Dropout(0.5),                 
                    Conv2D(filters=64,  kernel_size=(3,3), activation="relu"),
                    MaxPooling2D(2,2, padding='same'),
                    Dropout(0.5),         
                    Flatten(),             
                    Dense(units=25, activation="softmax"),
])
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
in_shape=(size,size, channels)
model=build_model(in_shape)   
model.summary() 

# Create candidates for HP tuning
learningrates = [0.0001, 0.001, 0.002]
batchsizes = [64, 128, 256, 512]

seed(1)
val_accuracy=[]
for lr in learningrates:
    for batch in batchsizes:
# Create model
        model_tune = build_model(in_shape=(28,28,1))
# Train the model
        history = model_tune.fit(X_train_cnn, y_train, epochs=1, batch_size=batch,validation_data=(X_val_cnn, y_val), verbose=1)
        model_tune.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
        train_accuracy = history.history['accuracy']
        val_accuracy += [history.history['val_accuracy']]
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        print('learning rate:{}, batch size:{}'.format(lr, batch))
highest_val_acc=max(val_accuracy)


seed(1)
model_CNN=build_model(in_shape)   

history=model_CNN.fit(X_train_cnn, y_train, validation_data=(X_val_cnn, y_val),epochs=100, batch_size=64, verbose=1)
model_CNN.compile(optimizer=keras.optimizers.Adam(learning_rate=0.002))
preds_cnn = np.argmax(model_CNN.predict(test_data_cnn),axis = 1) 

accuracy_test_cnn=accuracy_score(test_label, preds_cnn)

loss_test_cnn = model_CNN.evaluate(test_data_cnn)
print('Error on test data: {}'.format(loss_test_cnn))

#plot accuracy and loss of training and validation
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']


plt.figure(figsize=(18,6))
plt.subplot(1,2,1)
plt.title('Training and Validation accuracy by epoch', fontsize=16)
plt.plot(train_accuracy, label='Training accuracy')
plt.plot(val_accuracy, label='Validation accuracy')
plt.legend(['Train acc', 'Val acc'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1,2,2)
plt.title('Training and Validation loss by epoch', fontsize=16)
plt.plot(train_loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.legend(['Train loss', 'Val loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# plot the confusion matrix of CNN performance
CM_cnn= confusion_matrix(test_label, preds_cnn)
plt.figure(figsize = (15,15))
sns.heatmap(CM_cnn, annot=True, cmap="Blues", fmt = 'g', 
            xticklabels=np.unique(test_label), yticklabels=np.unique(test_label))
plt.xlabel("Predicted Classes")
plt.ylabel("True Classes")
plt.title("Confusion Matrix")
plt.show()

# recall per class:
labels=[0,1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
recall_cnn=[]
for t in labels:
    TP, FP, FN=class_scores(test_label.tolist(), preds_cnn.tolist(),t)
    recall_cnn+=[TP/(TP+FN)]

recall_dict_cnn={}
for l,r in zip(labels,recall_cnn):
    recall_dict_cnn[l]=r
print(recall_dict_cnn)

# find the letters that are easiest to classify (letters having recall score =1)
indices_easy_cnn = [i for i, x in enumerate(recall_cnn) if x == 1]
easy_cnn=[]
for ind in indices_easy_cnn:
    easy_cnn+=[labels[ind]]
easy_letters_cnn=set(easy_cnn)

# find the letter that is the most difficult to classify (letter having the lowest recall score)
indices_hard_cnn = [i for i, x in enumerate(recall_cnn) if x == min(recall_cnn)]
hard_cnn=[]
for ind in indices_hard_cnn:
    hard_cnn+=[labels[ind]]
hard_letters_cnn=set(hard_cnn)



## TASK 2--------------------------------------------------------------------
test= np.load("test_images_task2.npy") 

import cv2
import numpy as np

#load image into variable
img_rgb = cv2.imread('test_images_task2.npy')

#load template
template = cv2.imread('template.jpg')
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
h=sliding_window(test, 1, (28,28))
