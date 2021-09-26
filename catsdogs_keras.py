import time
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils import to_categorical
from keras.models import (
                          Model,
                          )
from keras.layers import (
                          Activation,
                          Conv2D,
                          Dense,
                          Flatten,
                          Input,
                          MaxPooling2D,
                          Dropout,
                          )
import keras
from keras.models import Sequential

# Required magic to display matplotlib plots in notebooks
#%matplotlib inline

# Set up a figure of an appropriate size
# create empty list
images_dataset = []
class_labels = []
batch_size = 128
epochs = 50

# Helper function to resize image proportionally. size is a tuple (height,width)
def resize_image(img, size): 
    from PIL import Image, ImageOps 
    
    # resize the image so the longest dimension matches our target size
    img.thumbnail(size, Image.ANTIALIAS)
    
    # Create a new square white background image
    newimg = Image.new("RGB", size, (255, 255, 255))
    
    # Paste the resized image into the center of the square background
    if np.array(img).shape[2] == 4:
        # If the source is in RGBA format, use a mask to eliminate the transparency
        newimg.paste(img, (int((size[0] - img.size[0]) / 2), int((size[1] - img.size[1]) / 2)), mask=img.split()[3])
    else:
        newimg.paste(img, (int((size[0] - img.size[0]) / 2), int((size[1] - img.size[1]) / 2)))
  
    # return the resized image
    return newimg

# loop through the subfolders in the input directory
image_base_dir = './train'
n = 0
for root, folders, filenames in os.walk(image_base_dir):
    for folder in folders:
        class_labels.append(folder)
        print('processing folder:{}'.format(folder))
        n = n+1
        files = os.listdir(os.path.join(root,folder))
        for file in files:
            # construct the fully image filename(including path)
            imgFile = os.path.join(image_base_dir,folder, file)
            # read image data
            img = Image.open(imgFile)
            # resize image
            img = resize_image(img,(128,128))
            # images_dataset is list of numpy array and label
            images_dataset.append([np.array(np.array(img)),folder])

print('n={}'.format(n))
# shuffle the dataset
np.random.shuffle(images_dataset)

# fetch image data from gear_dataset
images_data = list(map(
                       lambda item: item[0],
                       images_dataset
                       ))

# fetch label data from gear_dataset
images_labels = list(map(
                       lambda item: item[1],
                       images_dataset,
                       ))
# images_data and images_labels are list now, should be converted to np.array
images_data = np.array(images_data,dtype=np.float)/255.
images_labels = np.array(images_labels)

# one hot encoding labels
# encode the labels into one hot encoding
encoder = preprocessing.LabelEncoder()
encoder.fit(images_labels)
print('class={}'.format(encoder.classes_))   # ['cats' 'dogs']
le_classes = encoder.transform(encoder.classes_)  # [0,1]
# zip pack the two object's element into a tuple: [0:'cats,1:'dogs']
# create a dictionary from a list of tuple
le_name_mapping = dict(zip(le_classes,encoder.classes_))
print(le_name_mapping,le_name_mapping[0])
output2 = open('pzsmodel2_labdic.pkl', 'wb')
pickle.dump(le_name_mapping, output2)
output2.close()

# save the digits and its corresponding labels
transfomed_label = encoder.fit_transform(class_labels)
print(transfomed_label)

# string to number. cats->0 dogs->1
labels_id = encoder.transform(images_labels)
# one hot encoding. Converts a class vector (integers) to binary class matrix. A binary matrix representation of the input. The classes axis is placed last.
labels_encoded = to_categorical(labels_id)
# split training and testing dataset
train_x,test_x,train_y,test_y = train_test_split(images_data,labels_encoded,test_size=0.2)
#print('shape={}'.format(np.array(images_dataset).shape))

# 
model = Sequential()
# strides=(1,1) padding='valid'
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=(128,128,3)))
print('output shape1={}'.format(model.output_shape))
model.add(Dropout(0.2))
print('output shape2={}'.format(model.output_shape))
# pool_size: integer or tuple of 2 integers, factors by which to downscale (vertical, horizontal). 
# strides: Integer, tuple of 2 integers, or None. Strides values. If None, it will default to pool_size.
model.add(MaxPooling2D(pool_size=(3, 3)))
print('output shape3={}'.format(model.output_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
print('output shape4={}'.format(model.output_shape))
model.add(MaxPooling2D(pool_size=(3, 3)))
print('output shape5={}'.format(model.output_shape))
model.add(Dropout(0.2))
model.add(Conv2D(128, (3, 3), activation='relu'))
print('output shape6={}'.format(model.output_shape))
model.add(MaxPooling2D(pool_size=(3, 3)))
print('output shape7={}'.format(model.output_shape))
model.add(Flatten())
print('output shape8={}'.format(model.output_shape))
model.add(Dense(128, activation='relu'))
# Softmax makes the output sum up to 1 so the output can be interpreted as probabilities.
model.add(Dense(n, activation='softmax'))
print('output shape9={}'.format(model.output_shape))
# Configures the model for training.
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


# Trains the model for a given number of epochs (iterations on a dataset).
history= model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_x, test_y))

# Returns the loss value & metrics values for the model in test mode.
score = model.evaluate(test_x, test_y, verbose=0)
print('score={}'.format(score))
#print('history={}'.format(history.accuracy))

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
                                 # batch_size=128,
                                 )
# predicted_labels_encoded is an array ,so np.argmax must specify the axis, so the result is also an arrar                                 
predicted_labels = np.argmax(predicted_labels_encoded,axis=1) 
print('predicted digits={}'.format(predicted_labels))                                
predicted_labels = encoder.inverse_transform(predicted_labels)
print('predicted labels=',predicted_labels)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

i = 0
#fig = plt.figure(figsize=(10, 10))
for data in predict_data:
    ax = plt.subplot(1,4,i+1)
    #img = Image.open(test_img_url) 
    #img = np.array(img)
    data = (data * 255.).astype(np.uint8)
    imgplot = plt.imshow(data)
    ax.set_title(predicted_labels[i])
    i = i+1
plt.show()
# save the model

model.save('catsdogs.h5')
