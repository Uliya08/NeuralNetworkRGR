import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import img_to_array

# Создаем папку для сохранения весов
os.makedirs('model_weights', exist_ok=True)

# Загрузка и подготовка данных
data = []
labels = []
classes = len(os.listdir('train'))
for num in range(0, classes):
    path = os.path.join('train', str(num))
    imagePaths = os.listdir(path)
    for img in imagePaths:
        image = Image.open(os.path.join(path, img))
        image = image.resize((30, 30))
        image = img_to_array(image)
        data.append(image)
        labels.append(num)

data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Функция для подсчета количества изображений по классам
def cnt_img_in_classes(labels):
    count = {}
    for i in labels:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    return count

# Визуализация распределения классов
samples_distribution = cnt_img_in_classes(y_train)

def diagram(count_classes):
    plt.bar(range(len(count_classes)), sorted(list(count_classes.values())), align='center')
    plt.xticks(range(len(count_classes)), sorted(list(count_classes.keys())), rotation=90, fontsize=7)
    plt.show()

diagram(samples_distribution)

# Аугментация данных
def aug_images(images, p):
    from imgaug import augmenters as iaa
    augs = iaa.SomeOf((2, 4),
                      [
                          iaa.Crop(px=(0, 4)),
                          iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
                          iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                          iaa.Affine(rotate=(-45, 45)),
                          iaa.Affine(shear=(-10, 10))
                      ])

    seq = iaa.Sequential([iaa.Sometimes(p, augs)])
    res = seq.augment_images(images)
    return res

def augmentation(images, labels):
    min_imgs = 500
    classes = cnt_img_in_classes(labels)
    for i in range(len(classes)):
        if (classes[i] < min_imgs):
            add_num = min_imgs - classes[i]
            imgs_for_augm = []
            lbls_for_augm = []
            for j in range(add_num):
                im_index = random.choice(np.where(labels == i)[0])
                imgs_for_augm.append(images[im_index])
                lbls_for_augm.append(labels[im_index])
            augmented_class = aug_images(imgs_for_augm, 1)
            augmented_class_np = np.array(augmented_class)
            augmented_lbls_np = np.array(lbls_for_augm)
            images = np.concatenate((images, augmented_class_np), axis=0)
            labels = np.concatenate((labels, augmented_lbls_np), axis=0)
    return (images, labels)

X_train, y_train = augmentation(X_train, y_train)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

augmented_samples_distribution = cnt_img_in_classes(y_train)
diagram(augmented_samples_distribution)

# Преобразование меток в one-hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Архитектура модели
class Net:
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (height, width, depth)
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=inputShape))
        model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(classes, activation='softmax'))
        return model

# Нормализация данных
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Проверка размерностей
print("Проверка размеров:")
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)

# Создание и компиляция модели
model = Net.build(width=30, height=30, depth=3, classes=43)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Callback для сохранения весов
checkpoint = ModelCheckpoint(
    filepath='model_weights/weights-{epoch:02d}-{val_accuracy:.2f}.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,
    mode='max',
    verbose=1
)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
# Обучение модели с сохранением весов
epochs = 1
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stopping],  # Передаем как список
    verbose=1
)

# Сохранение финальных весов
model.save_weights('model_weights/final_weights.h5')

# Сохранение полной модели (опционально)
model.save('model_weights/full_model.h5')

plt.style.use("ggplot")  # Исправлено: "ggplot" вместо "plot"
plt.figure(figsize=(10, 5))

epochs = len(history.history['loss'])  # Берем реальное количество эпох

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.grid(True)
plt.show()
