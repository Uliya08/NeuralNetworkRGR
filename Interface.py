#!/usr/bin/python3
# -*- coding: utf-8 -*-

from customtkinter import *
import tkinter.filedialog as filedialog
from PIL import Image, ImageTk
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

set_appearance_mode('dark')
set_default_color_theme('blue')

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

class SignRecognizer:
    def __init__(self, model_path='model_weights/final_model.h5'):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """Загружает модель Keras"""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Файл модели {model_path} не найден")
            model = load_model(model_path)
            print("Модель успешно загружена")
            return model
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return None

    def load_and_prepare_image(self, image_path, target_size=(64, 64)):
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
            return None

    def predict_image(self, image_path):
        """Делает предсказание для одного изображения"""
        if self.model is None:
            return None, None, None

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
            return None, None, None


# Создание главного окна
root = CTk()
root.title('Распознавание дорожных знаков')
root.geometry('800x600')

# Инициализация распознавателя
recognizer = SignRecognizer()


def click_handler():
    """Обработчик нажатия кнопки для распознавания изображения"""
    # Выбор файла через диалоговое окно
    file_path = filedialog.askopenfilename(
        title="Выберите изображение дорожного знака",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
    )

    if not file_path:  # Если пользователь отменил выбор
        return

    # Получаем предсказание
    predicted_class, predicted_name, confidence = recognizer.predict_image(file_path)
    if predicted_name is None:
        result_label.configure(text="Ошибка распознавания!")
        return

    # Формируем текст результата
    result_text = (
        f"Предсказанный знак: {predicted_name}\n"
        f"Уверенность: {confidence:.2f}%\n"
    )

    # Обновляем метку с результатом
    result_label.configure(text=result_text)

    # Показываем изображение
    show_image(file_path)


def show_image(image_path):
    """Отображает выбранное изображение"""
    try:
        image = Image.open(image_path)
        image.thumbnail((400, 400))  # Уменьшаем изображение для отображения

        # Конвертируем для CTk
        photo = ImageTk.PhotoImage(image)

        # Если изображение уже отображалось, обновляем его
        if hasattr(root, 'image_label'):
            root.image_label.configure(image=photo)
            root.image_label.image = photo
        else:
            # Создаем новый label для изображения
            root.image_label = CTkLabel(root, image=photo, text="")
            root.image_label.image = photo
            root.image_label.place(relx=0.5, rely=0.75, anchor='center')
    except Exception as e:
        print(f"Ошибка отображения изображения: {e}")


# Создание кнопки
btn = CTkButton(
    master=root,
    text='Выбрать изображение',
    corner_radius=16,
    command=click_handler,
    font=("Arial", 14)
)
btn.place(relx=0.5, rely=0.2, anchor='center')

# Метка для вывода результата
result_label = CTkLabel(
    master=root,
    text="Здесь будет результат распознавания",
    font=("Arial", 16),
    wraplength=700,
    justify="left"
)
result_label.place(relx=0.5, rely=0.4, anchor='center')

# Запуск главного цикла
root.mainloop()