import tensorflow
import tensorflow.keras as keras
import numpy as np
# Train a CNN classifier
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
import requests
from io import BytesIO
import matplotlib.image as mpimg
import os
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


print (tensorflow.__version__)

from keras import backend as K

# Helper function
def predict_image(classifier, img):
    
    # Flatten the image data to correct feature format. its sjape is (1,128,128,3)
    imgfeatures = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    imgfeatures = imgfeatures.astype('float32')
    imgfeatures /= 255

    # Use the classifier to predict the class
    predicted_class = classifier.predict(imgfeatures)
    # return the index of the maximun value in an array 
    i = np.argmax(predicted_class, axis=1)
    return i
# Resize image to size proportinally. size is a tuple (width,height)
def resize_image(img, size):
    
    # Convert RGBA images to RGB
    if np.array(img).shape[2] == 4:
        img = img.convert('RGB')
        
    # resize the image
    img.thumbnail(size, Image.ANTIALIAS)
    # create a white background image
    newimg = Image.new("RGB", size, (255, 255, 255))
    # Pastes another image into this image.
    newimg.paste(img, (int((size[0] - img.size[0]) / 2), int((size[1] - img.size[1]) / 2)))
    
    return newimg


def load_data (folder):
    # iterate through folders, assembling feature, label, and classname data objects

    c = 0
    # create empty list
    features = []
    # create empty numpy array
    labels = np.array([])
    # create empty list
    classnames = []
    # os.walk return 3-tuple for each directory in rootfloder 
    # root: './train  dirs: all subdir in root, like ['cats','dogs'] , filenames is empty
    for root, dirs, filenames in os.walk(folder):    # root= ./train
        for d in dirs:
            # use the folder name as the class name for this label
            classnames.append(d)
            # return all files in specified directory. os.path.join(root,d) generates a full path of a directory
            files = os.listdir(os.path.join(root,d))
            print('classname={},c={}'.format(d,c))
            for f in files:
                imgFile = os.path.join(root,d, f)
#                print('file={}'.format(imgFile))
#                img = plt.imread(imgFile)   # img's shape is (128,128,3)
                # Image.open returns a PIL.Image object
                img = Image.open(imgFile)
                # resize image. 参数 size 是一个元组(tuple), (width,height)
                img = resize_image(img,(128,128))
                # convert Image data to numpy array. its shape is (128,128,3)
                img_arr = np.array(img)
                # append data to list
                features.append(img_arr)
                # 标签数值化。append data to numpy array. labels is a one dimention numpy array [0,0,...1,...]
                labels = np.append(labels, c)   # apend c to labels numpy array.  c is number 0,1,...
            c = c + 1
    features = np.array(features)
    # features 及 labels 一一对应,classnames 为类别标签
    return features, labels, classnames

print('current dir=',os.getcwd())
# Prepare the image data
# features in shape (8000,128,128,3), labels in shape (8000,) classnames is  a list ['cats','dogs']
features, labels, classnames = load_data('./train')
print('feature shape={}',format(features.shape))


# split into training and testing sets
# arrays : sequence of indexables with same length / shape[0] 
# Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.30)

# Format features. convert from int to float
x_train = x_train.astype('float32')
# RGB data normalization
x_train /= 255.
x_test = x_test.astype('float32')
x_test /= 255.

# labels one-hot encoding. before convertion: y_train in shape (5600,) that is [0,1,0,1,...] after concertion: y_train in shape (5600,2) that is [[1,0],[0,1],[1,0],...]
y_train = np_utils.to_categorical(y_train, len(classnames))
y_train = y_train.astype('float32')
y_test = np_utils.to_categorical(y_test, len(classnames))
y_test = y_test.astype('float32')

model = Sequential()
# filter size (3,3) step default is (1,1)
model.add(Conv2D(32, (3, 3), input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]), activation='relu'))
print('output shape={}'.format(model.output_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
print('output shape={}'.format(model.output_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
print('output shape={}'.format(model.output_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
print('output shape={}'.format(model.output_shape))
model.add(Dropout(0.5))
print('output shape={}'.format(model.output_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
print('output shape={}'.format(model.output_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
print('output shape={}'.format(model.output_shape))
model.add(Dropout(0.5))
print('output shape={}'.format(model.output_shape))
model.add(Flatten())
print('output shape={}'.format(model.output_shape))
#model.add(Dense(3000, activation='relu'))
#print('output shape={}'.format(model.output_shape))
#model.add(Dense(1000, activation='relu'))
#print('output shape={}'.format(model.output_shape))
#model.add(Dense(100, activation='relu'))
#print('output shape={}'.format(model.output_shape))
model.add(Dense(len(classnames), activation='softmax'))
print('output shape={}'.format(model.output_shape))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
num_epochs = 25
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=64, validation_data=(x_test, y_test))

'''
epoch_nums = range(1,num_epochs+1)
training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()
'''
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# load predict images
predict_data = []
predict_images = []
size = (128,128)
files = os.listdir('./predict')
for f in files:
    if not 'jpg' in f:
        continue
    img = Image.open(os.path.join('./predict',f))
    # resize images
    img = resize_image(img,size)
    predict_data.append(np.array(img))


#img = Image.open('resized_images/hardshell_jackets_test/resized_10269570x1012905_zm.jpeg')
predict_data = np.array(predict_data,dtype='float')/255.

predicted_labels_encoded = model.predict(
                                 predict_data,
                                 )
# predicted_labels_encoded is an array ,so np.argmax must specify the axis, so the result is also an arrar                                 
predicted_labels = np.argmax(predicted_labels_encoded,axis=1) 
print('predicted digits={}'.format(predicted_labels))                                
#predicted_labels = encoder.inverse_transform(predicted_labels)
#print('predicted labels=',predicted_labels)
i = 0
#fig = plt.figure(figsize=(10, 10))
for data in predict_data:
    ax = plt.subplot(1,4,i+1)
    # image data should be converted back to its original value in order to show image
    data = (data * 255.).astype(np.uint8)
    imgplot = plt.imshow(data)
    # classnames is a dictionary [0:'cats',1:'dogs']
    ax.set_title(classnames[predicted_labels[i]])
    i = i+1
plt.show()
