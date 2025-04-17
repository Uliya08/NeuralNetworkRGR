import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras import layers, models, callbacks, utils
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
import tensorflow as tf
from imgaug import augmenters as iaa

# Настройки
os.makedirs('model_weights', exist_ok=True)
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 12
PATIENCE = 5
NUM_CLASSES = 43
MIN_SAMPLES = 800  # Минимальное количество образцов на класс


def create_augmenter():
    """Создает аугментатор с базовыми преобразованиями"""
    return iaa.Sequential([
        iaa.Fliplr(0.3),
        iaa.Affine(
            rotate=(-15, 15),
            scale=(0.9, 1.1),
            translate_percent=(-0.1, 0.1)
        ),
        iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0, 0.5))),
        iaa.Sometimes(0.2, iaa.AdditiveGaussianNoise(scale=(0, 0.03 * 255)))
    ])


def load_and_preprocess_data(data_dir='train'):
    """Загрузка и предварительная обработка данных"""
    data, labels = [], []

    for class_id in range(NUM_CLASSES):
        class_dir = os.path.join(data_dir, str(class_id))
        if not os.path.exists(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            try:
                img = Image.open(os.path.join(class_dir, img_name)).convert('RGB')
                img = img.resize(IMG_SIZE)
                data.append(img_to_array(img))
                labels.append(class_id)
            except Exception as e:
                print(f"Ошибка загрузки {img_name}: {e}")

    return np.array(data), np.array(labels)


def build_optimized_model(input_shape, num_classes):
    """Оптимизированная архитектура модели"""
    model = models.Sequential([
        # Блок 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Блок 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Блок 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Полносвязные слои
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall')])
    return model


def main():
    # Загрузка данных
    print("[INFO] Загрузка и предобработка данных...")
    data, labels = load_and_preprocess_data()

    # Балансировка классов
    unique, counts = np.unique(labels, return_counts=True)
    for class_id, count in zip(unique, counts):
        if count < MIN_SAMPLES:
            print(f"Класс {class_id} имеет только {count} образцов")

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels)

    # Преобразование меток
    y_train = utils.to_categorical(y_train, NUM_CLASSES)
    y_test = utils.to_categorical(y_test, NUM_CLASSES)

    # Нормализация
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Построение модели
    print("[INFO] Создание модели...")
    model = build_optimized_model((*IMG_SIZE, 3), NUM_CLASSES)
    model.summary()

    # Коллбэки
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        callbacks.ModelCheckpoint(
            filepath='model_weights/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            verbose=1,
            min_lr=1e-6
        )
    ]

    # Обучение модели
    print("[INFO] Обучение модели...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks_list,
        verbose=1
    )

    # Сохранение модели
    model.save('model_weights/final_model.h5')
    print("[INFO] Модель сохранена")

    # Визуализация результатов
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss Curves')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy Curves')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

    # Оценка модели
    print("[INFO] Оценка модели...")
    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {results[1]:.4f}")
    print(f"Test Precision: {results[2]:.4f}")
    print(f"Test Recall: {results[3]:.4f}")


if __name__ == "__main__":
    main()