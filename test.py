import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import random
import tensorflow as tf

# Уменьшаем уровень логов TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Словарь русских названий дорожных знаков (сокращенный под ваш CSV)
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
    """Загружает аннотации из CSV-файла"""
    try:
        df = pd.read_csv(csv_file)
        annotations = {}

        for _, row in df.iterrows():
            filename = os.path.basename(row['Path'])
            class_id = int(row['ClassId'])
            annotations[filename] = {
                'bbox': [int(row['Roi.X1']), int(row['Roi.Y1']),
                         int(row['Roi.X2']), int(row['Roi.Y2'])],
                'class_id': class_id,
                'true_name': RUSSIAN_SIGN_NAMES.get(class_id, f"Неизвестный класс ({class_id})")
            }
        print(f"Загружено {len(annotations)} аннотаций")
        return annotations
    except Exception as e:
        print(f"Ошибка загрузки аннотаций: {str(e)}")
        return None


def load_and_prepare_image(image_path, target_size=(64, 64)):
    """Загружает и подготавливает изображение для модели"""
    try:
        image = Image.open(image_path).convert('RGB').resize(target_size)
        return np.expand_dims(img_to_array(image) / 255.0, axis=0)
    except Exception as e:
        print(f"Ошибка обработки изображения {image_path}: {str(e)}")
        return None


def predict_and_visualize(model, annotations, test_dir='Test'):
    """Делает предсказание и визуализирует результат"""
    if not annotations:
        return

    available_images = [f for f in os.listdir(test_dir) if f in annotations]
    if not available_images:
        return

    filename = random.choice(available_images)
    image_path = os.path.join(test_dir, filename)
    image_array = load_and_prepare_image(image_path)

    if image_array is None:
        return

    try:
        prediction = model.predict(image_array)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        annotation = annotations[filename]

        plt.figure(figsize=(10, 8))
        plt.imshow(Image.open(image_path))

        # Рисуем bounding box
        x1, y1, x2, y2 = annotation['bbox']
        plt.gca().add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor='green', facecolor='none'
        ))

        plt.title(
            f"Файл: {filename}\n"
            f"Истинный класс: {annotation['true_name']}\n"
            f"Предсказанный класс: {RUSSIAN_SIGN_NAMES.get(predicted_class, 'Неизвестный класс')}\n"
            f"Уверенность: {confidence:.2f}%",
            fontsize=12, pad=20
        )
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Ошибка: {str(e)}")


def main():
    print("[INFO] Загрузка аннотаций...")
    annotations = load_annotations()

    if not annotations:
        print("[ERROR] Не удалось загрузить аннотации!")
        return

    print("[INFO] Загрузка модели...")
    try:
        model = load_model('model_weights/final_model.h5')
    except Exception as e:
        print(f"[ERROR] Ошибка загрузки модели: {str(e)}")
        return

    for i in range(3):
        print(f"\nТест #{i + 1}")
        predict_and_visualize(model, annotations)


if __name__ == "__main__":
    main()