import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import random

# Словарь русских названий дорожных знаков
RUSSIAN_SIGN_NAMES = {
    0: "Ограничение скорости (20 км/ч)",
    1: "Ограничение скорости (30 км/ч)",
    2: "Ограничение скорости (50 км/ч)",
    3: "Ограничение скорости (60 км/ч)",
    4: "Ограничение скорости (70 км/ч)",
    5: "Ограничение скорости (80 км/ч)",
    6: "Конец ограничения скорости (80 км/ч)",
    7: "Ограничение скорости (100 км/ч)",
    8: "Ограничение скорости (120 км/ч)",
    9: "Обгон запрещен",
    10: "Обгон грузовикам запрещен",
    11: "Перекресток с второстепенной дорогой",
    12: "Главная дорога",
    13: "Уступите дорогу",
    14: "Стоп",
    15: "Движение запрещено",
    16: "Движение грузовиков запрещено",
    17: "Въезд запрещен",
    18: "Опасность",
    19: "Опасный поворот налево",
    20: "Опасный поворот направо",
    21: "Несколько поворотов",
    22: "Неровная дорога",
    23: "Скользкая дорога",
    24: "Сужение дороги справа",
    25: "Дорожные работы",
    26: "Светофорное регулирование",
    27: "Пешеходный переход",
    28: "Дети",
    29: "Велодорожка",
    30: "Снег/лед",
    31: "Дикие животные",
    32: "Конец всех ограничений",
    33: "Поворот направо обязателен",
    34: "Поворот налево обязателен",
    35: "Движение прямо обязательно",
    36: "Движение прямо или направо",
    37: "Движение прямо или налево",
    38: "Держаться справа",
    39: "Держаться левой стороны",
    40: "Круговое движение",
    41: "Конец зоны запрета обгона",
    42: "Конец зоны запрета обгона грузовикам"
}


def load_annotations(csv_file='Test.csv'):
    """Загружает аннотации из CSV-файла, пропуская заголовок"""
    annotations = {}
    try:
        # Читаем CSV с помощью pandas (автоматически пропустит заголовок)
        df = pd.read_csv(csv_file, header=0)

        for _, row in df.iterrows():
            filename = row[7].split('/')[-1]  # Получаем только имя файла
            annotations[filename] = {
                'bbox': [int(row[0]), int(row[1]), int(row[2]), int(row[3])],
                'center': [int(row[4]), int(row[5])],
                'class_id': int(row[6]),
                'true_name': RUSSIAN_SIGN_NAMES.get(int(row[6]), f"Неизвестный класс ({int(row[6])})")
            }
        print(f"Загружено {len(annotations)} аннотаций")
    except Exception as e:
        print(f"Ошибка загрузки аннотаций: {e}")
    return annotations


def load_and_prepare_image(image_path, target_size=(30, 30)):
    try:
        image = Image.open(image_path)
        image = image.resize(target_size)
        image_array = img_to_array(image)
        return np.expand_dims(image_array, axis=0) / 255.0
    except Exception as e:
        print(f"Ошибка загрузки изображения {image_path}: {e}")
        return None


def predict_and_visualize(model, annotations, test_dir='test'):
    if not annotations:
        print("Нет данных аннотаций!")
        return

    # Выбираем случайное изображение из аннотаций
    filename = random.choice(list(annotations.keys()))
    annotation = annotations[filename]

    # Загружаем и обрабатываем изображение для модели
    image_path = os.path.join(test_dir, filename)
    image_array = load_and_prepare_image(image_path)

    if image_array is None:
        return

    # Делаем предсказание
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    predicted_name = RUSSIAN_SIGN_NAMES.get(predicted_class, f"Неизвестный класс ({predicted_class})")
    confidence = np.max(prediction) * 100

    # Визуализация
    plt.figure(figsize=(6, 5))

    # Показываем оригинальное изображение
    original_image = Image.open(image_path)
    plt.imshow(original_image)

    # Рисуем bounding box из аннотации
    x1, y1, width, height = annotation['bbox']
    rect = plt.Rectangle((x1, y1), width, height,
                         linewidth=2, edgecolor='green', facecolor='none')
    plt.gca().add_patch(rect)

    # Добавляем метки
    title = (f"Файл: {filename}\n"
             f"Настоящий класс: {annotation['true_name']} (класс {annotation['class_id']})\n"
             f"Предсказанный класс: {predicted_name} (класс {predicted_class})\n"
             f"Уверенность: {confidence:.2f}%")

    plt.title(title, fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Уменьшаем уровень логов TensorFlow

    # Загрузка аннотаций
    print("[INFO] Загрузка аннотаций...")
    annotations = load_annotations('Test.csv')

    if not annotations:
        print("Не удалось загрузить аннотации!")
        return

    # Загрузка модели
    print("[INFO] Загрузка модели...")
    try:
        model = load_model('model_weights/full_model.h5')  # Или ваш путь к модели
        print("Модель успешно загружена")
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    # Проверка нескольких случайных изображений
    for _ in range(3):  # Можно изменить количество
        print("\n[INFO] Анализ изображения...")
        predict_and_visualize(model, annotations)


if __name__ == "__main__":
    main()

