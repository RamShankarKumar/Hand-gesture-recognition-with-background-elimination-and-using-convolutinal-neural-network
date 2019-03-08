

# In[1]:
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import cv2
from sklearn.utils import shuffle


# In[7]:


#Load Images from one
loadedImages = []
for i in range(1, 1201):
    image = cv2.imread('E:\\paper topic\\data set\\train_image\\one\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#Load Images From two
for i in range(1, 1201):
    image = cv2.imread('E:\\paper topic\\data set\\train_image\\two\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))
    
#Load Images From three
for i in range(1, 1201):
    image = cv2.imread('E:\\paper topic\\data set\\train_image\\three\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))
    
#load images from four
for i in range(1, 1201):
    image = cv2.imread('E:\\paper topic\\data set\\train_image\\four\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#load images from five
for i in range(1, 1201):
    image = cv2.imread('E:\\paper topic\\data set\\train_image\\five\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#load images from fist
for i in range(1, 1201):
    image = cv2.imread('E:\\paper topic\\data set\\train_image\\fist\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#load images from L
for i in range(1, 1201):
    image = cv2.imread('E:\\paper topic\\data set\\train_image\\L\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#load images from swing
for i in range(1, 1201):
    image = cv2.imread('E:\\paper topic\\data set\\train_image\\swing\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#load images from palm
for i in range(1, 1201):
    image = cv2.imread('E:\\paper topic\\data set\\train_image\\palm\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#load images from rock on
for i in range(1, 1201):
    image = cv2.imread('E:\\paper topic\\data set\\train_image\\rock on\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))

#load images from blank
for i in range(1, 1201):
    image = cv2.imread('E:\\paper topic\\data set\\train_image\\blank\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    loadedImages.append(gray_image.reshape(89, 100, 1))    
    
    
# In[8]:


# Create OutputVector

outputVectors = []
for i in range(1, 1201):
    outputVectors.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

for i in range(1, 1201):
    outputVectors.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

for i in range(1, 1201):
    outputVectors.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    
for i in range(1, 1201):
    outputVectors.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    
for i in range(1, 1201):
    outputVectors.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    
for i in range(1, 1201):
    outputVectors.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    
for i in range(1, 1201):
    outputVectors.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    
for i in range(1, 1201):
    outputVectors.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    
for i in range(1, 1201):
    outputVectors.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    
for i in range(1, 1201):
    outputVectors.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    
for i in range(1, 1201):
    outputVectors.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


# In[9]:


testImages = []

#Load Images for one
for i in range(1, 201):
    image = cv2.imread('E:\\paper topic\\data set\\test_image\\one\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

#Load Images for two
for i in range(1, 201):
    image = cv2.imread('E:\\paper topic\\data set\\test_image\\two\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
    
#Load Images for three
for i in range(1, 201):
    image = cv2.imread('E:\\paper topic\\data set\\test_image\\three\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
    
#load image for four
for i in range(1, 201):
    image = cv2.imread('E:\\paper topic\\data set\\test_image\\four\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
    
#load image for five
for i in range(1, 201):
    image = cv2.imread('E:\\paper topic\\data set\\test_image\\five\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
    
#load image for fist
for i in range(1, 201):
    image = cv2.imread('E:\\paper topic\\data set\\test_image\\fist\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
    
#load image for L
for i in range(1, 201):
    image = cv2.imread('E:\\paper topic\\data set\\test_image\\L\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))

#load images for swing
for i in range(1, 201):
    image = cv2.imread('E:\\paper topic\\data set\\test_image\\swing\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
    
#load images for palm
for i in range(1, 201):
    image = cv2.imread('E:\\paper topic\\data set\\test_image\\palm\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
    
#load images for rock on
for i in range(1, 201):
    image = cv2.imread('E:\\paper topic\\data set\\test_image\\rock on\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
    
#load images for blank
for i in range(1, 201):
    image = cv2.imread('E:\\paper topic\\data set\\test_image\\blank\\' + str(i) + '.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    testImages.append(gray_image.reshape(89, 100, 1))
    

testLabels = []

for i in range(1, 201):
    testLabels.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    
for i in range(1, 201):
    testLabels.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

for i in range(1, 201):
    testLabels.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    
for i in range(1, 201):
    testLabels.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    
for i in range(1, 201):
    testLabels.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    
for i in range(1, 201):
    testLabels.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    
for i in range(1, 201):
    testLabels.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    
for i in range(1, 201):
    testLabels.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    
for i in range(1, 201):
    testLabels.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    
for i in range(1, 201):
    testLabels.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    
for i in range(1, 201):
    testLabels.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])


# In[10]:


# Define the CNN Model
tf.reset_default_graph()
convnet=input_data(shape=[None,89,100,1],name='input')
convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet,1200,activation='relu')
convnet=dropout(convnet,0.75)

convnet=fully_connected(convnet,11,activation='softmax')

convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(convnet,tensorboard_verbose=0)


# In[11]:


# Shuffle Training Data
loadedImages, outputVectors = shuffle(loadedImages, outputVectors, random_state=0)

# Train model
model.fit(loadedImages, outputVectors, n_epoch=100,
           validation_set = (testImages, testLabels),
           snapshot_step=100, show_metric=True, run_id='convnet_coursera')

model.save("E:\\paper topic\\Hand-Gesture-Recognition-Using-Background-Elllimination-and-Convolution-Neural-Network-master\\epoch10GestureRecogModel.tfl")


