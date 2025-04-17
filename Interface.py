from customtkinter import *
import tkinter.filedialog as filedialog
from MyTest import RoadSignRecognizer
from tensorflow import keras
import os

# Настройка темы (темный/светлый режим + цветовая схема)
set_appearance_mode('dark')
set_default_color_theme('blue')

# Создание главного окна
root = CTk()
root.title('Распознавание дорожных знаков')
root.geometry('800x600')

# Инициализация распознавателя
recognizer = RoadSignRecognizer(model_path='model_weights/best_model.h5')


def click_handler():
    """Обработчик нажатия кнопки для распознавания изображения"""
    # Выбор файла через диалоговое окно
    file_path = filedialog.askopenfilename(
        title="Выберите изображение дорожного знака",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
    )

    if not file_path:  # Если пользователь отменил выбор
        return

    # Получаем имя файла для аннотаций
    filename = os.path.basename(file_path)

    # Получаем предсказание
    predicted_class, predicted_name, confidence = recognizer.predict_image(file_path)
    if predicted_name is None:
        result_label.configure(text="Ошибка распознавания!")
        return

    # Получаем реальный класс (если есть аннотации)
    true_class, true_name = recognizer.get_true_class(filename)

    # Формируем текст результата
    result_text = (
        f"Предсказанный знак: {predicted_name}\n"
        f"Уверенность: {confidence:.2f}%\n"
    )

    if true_name:
        result_text += f"Реальный знак: {true_name}"

    # Обновляем метку с результатом
    result_label.configure(text=result_text)

    # Показываем визуализацию (опционально)
    recognizer.visualize_prediction(file_path, predicted_name, confidence, true_name)


# Создание кнопки
btn = CTkButton(
    master=root,
    text='Выбрать изображение',
    corner_radius=16,
    command=click_handler,
    font=("Arial", 14)
)
btn.place(relx=0.5, rely=0.3, anchor='center')

# Метка для вывода результата
result_label = CTkLabel(
    master=root,
    text="Здесь будет результат распознавания",
    font=("Arial", 16),
    wraplength=700,
    justify="left"
)
result_label.place(relx=0.5, rely=0.6, anchor='center')

# Настройка окна поверх других
root.lift()
root.attributes('-topmost', True)
root.after_idle(root.attributes, '-topmost', False)

root.mainloop()