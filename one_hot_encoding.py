import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# 共有 5 个类别的标签
labels = [  0,
            1,
            0,
            2,          
            3,
            4
        ]
# after one-hot encoding
encoded_labels = [
            [1,
             0,
             0,       # --> 0
             0,
             0
            ],
            [0,
             1,
             0,       # --> 1
             0,
             0
            ],
            [1,
             0,
             0,       # --> 0
             0,
             0
            ],
            [0,
             0,
             1,       # --> 2
             0,
             0
            ],
            [0,
             0,
             0,       # --> 3
             1,
             0
            ],
            [0,
             0,
             0,       # --> 4
             0,
             1
            ]
]
# one-hot encoding
# to_categorical(y, num_classes=None). Converts a class vector (integers) to binary class matrix.
# y：number to be converted. num_classes：total number of classes.
en_labels = np_utils.to_categorical(labels, len(labels))

print('labels shape={}'.format(np.array(labels).shape))   # (6,)  we have 6 elements
print('encoded_labels shape={}'.format(np.array(encoded_labels).shape))   # (6,5)   we have 6 elements, 5 classes
print('en_labels={}'.format(en_labels))

# label strings
labels_str = ['cat','dog','sheep','cow','bull']
# convert to numbers
# encode the labels into one hot encoding
encoder = preprocessing.LabelEncoder()
encoder.fit(labels_str)
print('class={}'.format(encoder.classes_))   # ['bull' 'cat' 'cow' 'dog' 'sheep']
le_classes = encoder.transform(encoder.classes_)  # [0 1 2 3 4]
# 注意：encoder.classes_、le_classes 此时都已排序
print('le_classes={}'.format(le_classes))

