# -*- coding: utf-8 -*-

!pip install -q --upgrade ipython
!pip install -q --upgrade ipykernel

from google.colab import drive
drive.mount('/content/drive')

# load training data
import pandas as pd

train = pd.read_csv("/content/drive/My Drive/Train.csv")
train.head()

# number of classes
number_of_classes= train.Category.nunique()
print("Number of classes in the dataset: ", number_of_classes)

# convert the data into numpy arrays of two variables, x and y.
import numpy as np

x = np.array(train[['Id']])
y = np.array(train[['Category']])

print("Shape of Id Array:", x.shape) # Viewing the shape of X
print(x.dtype)
print("Shape of Category Array:", y.shape) # Viewing the shape of y
print(y.dtype)

# load image with its path from data file
import cv2 
images=[]

#range should be x.shape[0] but giving memory allocation error 
size = 2048
for i in range(size):
    dirname="/content/drive/My Drive/Data_SkinCancer/"
    file= str(x[i])
    file= file[file.find("[")+2:file.find("]")-1]
    filename=str(dirname)+str(file)+".jpg"
    image= cv2.imread(filename)
    images.append(image)
    #print(images[i].shape) #in order to show images have different sizes with 10 samples

len(images)

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('ME', ' Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis', 'Benign keratosis')
y_pos = np.arange(number_of_classes)
unique, counts = np.unique(y, return_counts=True)
dict(zip(unique, counts))
performance = [counts[0],counts[1],counts[2],counts[3],counts[4]]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of Samples')
plt.xlabel('Skin Cancer Types')
plt.title('Number of Samples per Class')

plt.show()

# showing that image sizes are different
#print(images[92].shape)
#print(images[9].shape)

# setting dim of the resize
height = 1024
width = 1024
dim = (width, height)
res_img = []
for i in range(size):  
    try:
      res = cv2.resize(images[i], dim, interpolation=cv2.INTER_LINEAR).astype('float32')
      res_img.append(res)
    except Exception as e:
      print("\n",i)

count=0
# Checcking the sizes
for i in range(len(res_img)):  
    count+= 1
    print("RESIZED", res_img[i].shape, i)
print(count)

from sklearn.model_selection import train_test_split

y= y[0:90]
x_train, x_test, y_train, y_test = train_test_split(res_img, y, test_size = 0.1)
x_train= np.array(x_train)
y_train.shape
x_test= np.array(x_test)
x_test.shape
y_test.shape
input_shape=(1024, 1024, 3)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
from keras.optimizers import Adam # I believe this is better optimizer for our case
from keras.preprocessing.image import ImageDataGenerator # to augmenting our images for increasing accuracy
from keras.utils.vis_utils import plot_model
import scipy
from sklearn.model_selection import train_test_split # to split our train data into train and validation sets
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(1) #

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(1024,1024,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(5, activation = "softmax")
])
model.compile(optimizer =  Adam() , loss = "categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    x_train,
    epochs=10,
    validation_data=y_test,
)
