import os
import classes
import matplotlib
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

data = []
labels = []

for num in range(0, classes):
    path = os.path.join('train',str(num))
    imagePaths = os.listdir(path)
    for img in imagePaths:
      image = Image.open(path + '/'+ img)
      image = image.resize((30,30))
      image = img_to_array(image)
      data.append(image)
      labels.append(num)

data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)
(39209, 30, 30, 3) (39209,)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
(31367, 30, 30, 3) (7842, 30, 30, 3) (31367,) (7842,)

def cnt_img_in_classes(labels):
    count = {}
    for i in labels:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    return count

samples_distribution = cnt_img_in_classes (y_train)

def diagram(count_classes):
    plt.bar(range(len(dct)), sorted(list(count_classes.values())), align='center')
    plt.xticks(range(len(dct)), sorted(list(count_classes.keys())), rotation=90, fontsize=7)
    plt.show()
diagram(samples_distribution)