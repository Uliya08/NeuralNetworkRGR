import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import random

# Разрешаем загрузку усеченных изображений
ImageFile.LOAD_TRUNCATED_IMAGES = True

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


class RoadSignRecognizer:
    def __init__(self, model_path='model_weights/full_model.h5', annotations_file='Test.csv'):
        self.model = self.load_model(model_path)
        self.annotations = self.load_annotations(annotations_file)
        self.root = tk.Tk()
        self.root.withdraw()

    def load_annotations(self, csv_file):
        """Загружает аннотации из CSV-файла"""
        if not os.path.exists(csv_file):
            print(f"Файл аннотаций {csv_file} не найден")
            return None

        try:
            df = pd.read_csv(csv_file, header=0)
            annotations = {}
            for _, row in df.iterrows():
                filename = row[7].split('/')[-1]
                annotations[filename] = {
                    'class_id': int(row[6]),
                    'true_name': RUSSIAN_SIGN_NAMES.get(int(row[6]), f"Неизвестный класс ({int(row[6])})")
                }
            print(f"Загружено {len(annotations)} аннотаций")
            return annotations
        except Exception as e:
            print(f"Ошибка загрузки аннотаций: {e}")
            return None

    def load_model(self, model_path):
        """Загружает модель Keras"""
        try:
            model = load_model(model_path)
            print("Модель успешно загружена")
            return model
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {e}")
            return None

    def get_true_class(self, filename):
        """Возвращает реальный класс из аннотаций"""
        if self.annotations and filename in self.annotations:
            return (self.annotations[filename]['class_id'],
                    self.annotations[filename]['true_name'])
        return None, None

    def load_and_prepare_image(self, image_path, target_size=(30, 30)):
        """Загружает и подготавливает изображение для модели"""
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize(target_size)
            image_array = img_to_array(image)
            return np.expand_dims(image_array, axis=0) / 255.0
        except Exception as e:
            print(f"Ошибка загрузки изображения {image_path}: {e}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")
            return None

    def predict_image(self, image_path):
        """Делает предсказание для одного изображения"""
        image_array = self.load_and_prepare_image(image_path)
        if image_array is None:
            return None, None, None

        try:
            prediction = self.model.predict(image_array)
            predicted_class = np.argmax(prediction)
            predicted_name = RUSSIAN_SIGN_NAMES.get(predicted_class,
                                                    f"Неизвестный класс ({predicted_class})")
            confidence = np.max(prediction) * 100
            return predicted_class, predicted_name, confidence
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            messagebox.showerror("Ошибка", f"Ошибка при распознавании: {e}")
            return None, None, None

    def visualize_prediction(self, image_path, predicted_name, confidence, true_name=None):
        """Визуализирует результат предсказания"""
        try:
            plt.figure(figsize=(10, 8))
            original_image = Image.open(image_path)
            plt.imshow(original_image)

            title = "Результат распознавания:\n"
            title += f"Предсказанный класс: {predicted_name}\n"
            title += f"Уверенность: {confidence:.2f}%\n"

            if true_name:
                title += f"\nРеальный класс: {true_name}"
            else:
                title += "\nРеальный класс: неизвестен"

            plt.title(title, fontsize=14, pad=20)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Ошибка визуализации: {e}")
            messagebox.showerror("Ошибка", f"Не удалось отобразить результат: {e}")

    def select_and_predict(self):
        """Открывает диалог выбора файла и делает предсказание"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение для распознавания",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )

        if not file_path:
            print("Файл не выбран")
            return

        filename = os.path.basename(file_path)
        print(f"\nАнализ изображения: {filename}")

        # Получаем предсказание
        predicted_class, predicted_name, confidence = self.predict_image(file_path)
        if predicted_name is None:
            return

        # Получаем реальный класс (если есть аннотации)
        true_class, true_name = self.get_true_class(filename)

        print(f"Результат: {predicted_name} (класс {predicted_class})")
        print(f"Уверенность: {confidence:.2f}%")
        if true_name:
            print(f"Реальный класс: {true_name} (класс {true_class})")

        # Показываем результат
        self.visualize_prediction(file_path, predicted_name, confidence, true_name)

    def run(self):
        """Основной цикл работы программы"""
        if not self.model:
            return

        while True:
            print("\n1. Загрузить свое изображение\n2. Выход")
            choice = input("Выберите действие: ").strip()

            if choice == '1':
                self.select_and_predict()
            elif choice == '2':
                print("Выход из программы")
                break
            else:
                print("Некорректный ввод")


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    try:
        recognizer = RoadSignRecognizer()
        recognizer.run()
    except ImportError as e:
        print(f"Ошибка: {e}\nУстановите tkinter для работы графического интерфейса")
        exit(1)