import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import json
import os
from datetime import datetime


class NeuralNetwork:
    """Нейронная сеть с обратным распространением ошибки"""

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        """
        Инициализация нейронной сети

        Args:
            input_size: количество входных нейронов (16*16 = 256)
            hidden_size: количество нейронов скрытого слоя (8)
            output_size: количество выходных нейронов (количество классов)
            learning_rate: норма обучения
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Инициализация весов случайными значениями
        # Веса между входным и скрытым слоем
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.5
        # Веса между скрытым и выходным слоем
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.5

        # Смещения
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        """Сигмоидальная функция активации"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        """Производная сигмоидальной функции"""
        return x * (1 - x)

    def forward(self, X):
        """
        Прямое распространение

        Args:
            X: входные данные

        Returns:
            output: выход сети
        """
        # Вход -> скрытый слой
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        # Скрытый слой -> выходной слой
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)

        return self.final_output

    def backward(self, X, y, output):
        """
        Обратное распространение ошибки

        Args:
            X: входные данные
            y: целевые значения
            output: выход сети
        """
        # Вычисление ошибки на выходном слое
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        # Вычисление ошибки на скрытом слое
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Обновление весов и смещений
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs):
        """
        Обучение сети

        Args:
            X: обучающие данные
            y: целевые значения
            epochs: количество эпох обучения

        Returns:
            losses: список значений ошибки для каждой эпохи
        """
        losses = []
        for epoch in range(epochs):
            # Прямое распространение
            output = self.forward(X)

            # Обратное распространение
            self.backward(X, y, output)

            # Вычисление ошибки
            loss = np.mean(np.square(y - output))
            losses.append(loss)

        return losses

    def predict(self, X):
        """
        Предсказание класса

        Args:
            X: входные данные

        Returns:
            Индекс класса с максимальной вероятностью
        """
        output = self.forward(X)
        return np.argmax(output, axis=1)

    def save_model(self, filename):
        """Сохранение модели в файл"""
        model_data = {
            'weights_input_hidden': self.weights_input_hidden.tolist(),
            'weights_hidden_output': self.weights_hidden_output.tolist(),
            'bias_hidden': self.bias_hidden.tolist(),
            'bias_output': self.bias_output.tolist(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate
        }
        with open(filename, 'w') as f:
            json.dump(model_data, f)

    def load_model(self, filename):
        """Загрузка модели из файла"""
        with open(filename, 'r') as f:
            model_data = json.load(f)

        self.weights_input_hidden = np.array(model_data['weights_input_hidden'])
        self.weights_hidden_output = np.array(model_data['weights_hidden_output'])
        self.bias_hidden = np.array(model_data['bias_hidden'])
        self.bias_output = np.array(model_data['bias_output'])
        self.input_size = model_data['input_size']
        self.hidden_size = model_data['hidden_size']
        self.output_size = model_data['output_size']
        self.learning_rate = model_data['learning_rate']


class PatternRecognitionApp:
    """Главное приложение для распознавания образов"""

    def __init__(self, root):
        self.root = root
        self.root.title("Распознавание арифметических операций")
        self.root.geometry("1100x800")

        # Параметры
        self.grid_size = 16
        self.cell_size = 40
        self.canvas_size = self.grid_size * self.cell_size

        # Матрица для хранения текущего рисунка
        self.grid = np.zeros((self.grid_size, self.grid_size))

        # Хардкод базовых паттернов арифметических операций
        self.base_patterns = self.create_base_patterns()

        # Нейронная сеть
        self.nn = None

        # Режим работы
        self.mode = tk.StringVar(value="draw")  # "draw" или "recognize"

        # Файлы для датасетов
        self.drawn_dataset_file = "drawn_dataset.json"
        self.generated_dataset_file = "generated_dataset.json"

        # Создание интерфейса
        self.create_widgets()

        # Загрузка датасетов если они существуют
        self.load_datasets()

    def create_base_patterns(self):
        """Создание базовых паттернов арифметических операций"""
        patterns = {}

        # Плюс (+)
        plus = np.zeros((16, 16))
        plus[7:9, 3:13] = 1  # горизонтальная линия
        plus[3:13, 7:9] = 1  # вертикальная линия
        patterns['+'] = plus

        # Минус (-)
        minus = np.zeros((16, 16))
        minus[7:9, 3:13] = 1  # горизонтальная линия
        patterns['-'] = minus

        # Умножение (*)
        multiply = np.zeros((16, 16))
        for i in range(4, 12):
            multiply[i, i] = 1  # диагональ \\
            multiply[i, 15 - i] = 1  # диагональ /
        multiply[7:9, 5:11] = 1  # горизонтальная
        multiply[5:11, 7:9] = 1  # вертикальная
        patterns['*'] = multiply

        # Деление (/)
        divide = np.zeros((16, 16))
        for i in range(3, 13):
            j = 15 - i
            if 0 <= j < 16:
                divide[i, j] = 1
        patterns['/'] = divide

        # Корень (√)
        sqrt = np.zeros((16, 16))
        sqrt[10:14, 3:5] = 1  # левая часть
        sqrt[7:11, 5:7] = 1  # средняя часть
        sqrt[3:8, 7:9] = 1  # правая часть вверх
        sqrt[3:5, 7:14] = 1  # горизонтальная часть
        patterns['V'] = sqrt

        # Процент (%)
        percent = np.zeros((16, 16))
        # Верхний кружок
        percent[3:6, 3:6] = 1
        percent[4, 4] = 0
        # Нижний кружок
        percent[10:13, 10:13] = 1
        percent[11, 11] = 0
        # Диагональ
        for i in range(3, 13):
            j = 15 - i
            if 0 <= j < 16:
                percent[i, j] = 1
        patterns['%'] = percent

        equal = np.zeros((16, 16))
        equal[7, 2:14] = 1
        equal[9, 2:14] = 1
        patterns['='] = equal

        # остальное без изменений
        return patterns

    def create_widgets(self):
        # Фрейм для холста (canvas) — первый столбец
        canvas_frame = ttk.LabelFrame(self.root, text="Область рисования 16x16", padding=10)
        canvas_frame.grid(row=0, column=0, padx=10, pady=10, rowspan=3, sticky='n')

        # Холст для рисования
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size,
                                bg='white', cursor='cross')
        self.canvas.pack()

        # Сетка
        for i in range(self.grid_size + 1):
            self.canvas.create_line(i * self.cell_size, 0,
                                    i * self.cell_size, self.canvas_size,
                                    fill='lightgray')
            self.canvas.create_line(0, i * self.cell_size,
                                    self.canvas_size, i * self.cell_size,
                                    fill='lightgray')

        # События мыши
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<Button-1>', self.paint)

        # ------ Второй столбец ------
        # Фрейм управления
        control_frame = ttk.LabelFrame(self.root, text="Управление", padding=10)
        control_frame.grid(row=0, column=1, padx=10, pady=10, sticky='n')

        ttk.Button(control_frame, text="Очистить холст",
                   command=self.clear_canvas).pack(fill='x', pady=5)

        ttk.Label(control_frame, text="Режим работы:").pack(pady=(10, 5))
        ttk.Radiobutton(control_frame, text="Рисование и сохранение",
                        variable=self.mode, value="draw").pack(anchor='w')
        ttk.Radiobutton(control_frame, text="Распознавание",
                        variable=self.mode, value="recognize").pack(anchor='w')

        ttk.Label(control_frame, text="Класс образа:").pack(pady=(10, 5))
        self.class_var = tk.StringVar(value="+")
        classes = ['+', '-', '*', '/', 'V', '%', '=']
        for cls in classes:
            ttk.Radiobutton(control_frame, text=cls,
                            variable=self.class_var, value=cls).pack(anchor='w')

        ttk.Button(control_frame, text="Сохранить рисунок",
                   command=self.save_drawn_pattern).pack(fill='x', pady=5)

        # Фрейм обучения
        train_frame = ttk.LabelFrame(self.root, text="Обучение сети", padding=10)
        train_frame.grid(row=1, column=1, padx=10, pady=10, sticky='n')

        ttk.Label(train_frame, text="Норма обучения:").pack()
        self.learning_rate_var = tk.DoubleVar(value=0.1)
        ttk.Entry(train_frame, textvariable=self.learning_rate_var, width=15).pack(pady=5)
        ttk.Label(train_frame, text="Количество эпох:").pack()
        self.epochs_var = tk.IntVar(value=1000)
        ttk.Entry(train_frame, textvariable=self.epochs_var, width=15).pack(pady=5)
        ttk.Button(train_frame, text="Генерировать датасет",
                   command=self.generate_dataset).pack(fill='x', pady=5)
        ttk.Button(train_frame, text="Обучить сеть",
                   command=self.train_network).pack(fill='x', pady=5)
        ttk.Button(train_frame, text="Распознать образ",
                   command=self.recognize_pattern).pack(fill='x', pady=5)

        # ------ Третий столбец ------
        dataset_frame = ttk.LabelFrame(self.root, text="Управление датасетами", padding=10)
        dataset_frame.grid(row=0, column=2, padx=10, pady=10, rowspan=2, sticky='n')

        ttk.Button(dataset_frame, text="Очистить нарисованные",
                   command=self.clear_drawn_dataset).pack(fill='x', pady=5)
        ttk.Button(dataset_frame, text="Очистить сгенерированные",
                   command=self.clear_generated_dataset).pack(fill='x', pady=5)
        ttk.Button(dataset_frame, text="Очистить оба датасета",
                   command=self.clear_all_datasets).pack(fill='x', pady=5)
        self.info_label = ttk.Label(dataset_frame, text="", justify='left')
        self.info_label.pack(pady=10)

        self.update_info()

    def paint(self, event):
        """Рисование на холсте"""
        x, y = event.x, event.y
        col = x // self.cell_size
        row = y // self.cell_size

        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            self.grid[row, col] = 1
            x1 = col * self.cell_size
            y1 = row * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill='black', outline='')

    def clear_canvas(self):
        """Очистка холста"""
        self.grid = np.zeros((self.grid_size, self.grid_size))
        self.canvas.delete('all')

        # Перерисовка сетки
        for i in range(self.grid_size + 1):
            self.canvas.create_line(i * self.cell_size, 0,
                                    i * self.cell_size, self.canvas_size,
                                    fill='lightgray')
            self.canvas.create_line(0, i * self.cell_size,
                                    self.canvas_size, i * self.cell_size,
                                    fill='lightgray')

    def save_drawn_pattern(self):
        """Сохранение нарисованного паттерна"""
        if np.sum(self.grid) == 0:
            messagebox.showwarning("Предупреждение", "Холст пуст!")
            return

        pattern_class = self.class_var.get()

        # Загрузка существующего датасета
        drawn_dataset = []
        if os.path.exists(self.drawn_dataset_file):
            with open(self.drawn_dataset_file, 'r') as f:
                drawn_dataset = json.load(f)

        # Добавление нового паттерна
        drawn_dataset.append({
            'pattern': self.grid.tolist(),
            'class': pattern_class,
            'timestamp': datetime.now().isoformat()
        })

        # Сохранение
        with open(self.drawn_dataset_file, 'w') as f:
            json.dump(drawn_dataset, f, indent=2)

        messagebox.showinfo("Успех", f"Паттерн '{pattern_class}' сохранен!")
        self.update_info()
        self.clear_canvas()

    def generate_dataset(self):
        """Генерация датасета на основе базовых паттернов с шумом"""
        samples_per_class = 20  # Количество образцов для каждого класса
        generated_dataset = []

        for class_name, base_pattern in self.base_patterns.items():
            for i in range(samples_per_class):
                # Добавление различных видов шума
                noisy_pattern = base_pattern.copy()

                # Случайный шум (несколько пикселей)
                noise_count = np.random.randint(1, 5)
                for _ in range(noise_count):
                    r, c = np.random.randint(0, 16, 2)
                    noisy_pattern[r, c] = 1 - noisy_pattern[r, c]

                # Небольшой сдвиг
                shift_x = np.random.randint(-1, 2)
                shift_y = np.random.randint(-1, 2)
                if shift_x != 0 or shift_y != 0:
                    noisy_pattern = np.roll(noisy_pattern, shift_x, axis=0)
                    noisy_pattern = np.roll(noisy_pattern, shift_y, axis=1)

                generated_dataset.append({
                    'pattern': noisy_pattern.tolist(),
                    'class': class_name,
                    'timestamp': datetime.now().isoformat()
                })

        # Сохранение
        with open(self.generated_dataset_file, 'w') as f:
            json.dump(generated_dataset, f, indent=2)

        messagebox.showinfo("Успех",
                            f"Сгенерировано {len(generated_dataset)} образцов!")
        self.update_info()

    def load_datasets(self):
        """Загрузка всех датасетов"""
        self.drawn_data = []
        self.generated_data = []

        if os.path.exists(self.drawn_dataset_file):
            with open(self.drawn_dataset_file, 'r') as f:
                self.drawn_data = json.load(f)

        if os.path.exists(self.generated_dataset_file):
            with open(self.generated_dataset_file, 'r') as f:
                self.generated_data = json.load(f)

    def train_network(self):
        """Обучение нейронной сети"""
        self.load_datasets()

        # Объединение датасетов
        all_data = self.drawn_data + self.generated_data

        if len(all_data) == 0:
            messagebox.showwarning("Предупреждение",
                                   "Нет данных для обучения! Создайте датасет.")
            return

        # Подготовка данных
        X = []
        y = []
        class_names = ['+', '-', '*', '/', 'V', '%', '=']

        for item in all_data:
            pattern = np.array(item['pattern']).flatten()
            X.append(pattern)

            # One-hot encoding
            class_idx = class_names.index(item['class'])
            y_vector = np.zeros(len(class_names))
            y_vector[class_idx] = 1
            y.append(y_vector)

        X = np.array(X)
        y = np.array(y)

        # Создание и обучение сети
        self.nn = NeuralNetwork(
            input_size=256,  # 16*16
            hidden_size=8,
            output_size=len(class_names),
            learning_rate=self.learning_rate_var.get()
        )

        epochs = self.epochs_var.get()
        losses = self.nn.train(X, y, epochs)

        messagebox.showinfo("Успех",
                            f"Обучение завершено!\nФинальная ошибка: {losses[-1]:.6f}")

    def recognize_pattern(self):
        if self.nn is None:
            messagebox.showwarning("Предупреждение", "Сначала обучите сеть!")
            return

        if np.sum(self.grid) == 0:
            messagebox.showwarning("Предупреждение", "Холст пуст!")
            return

        X = self.grid.flatten().reshape(1, -1)
        class_names = list(self.base_patterns.keys())
        prediction = self.nn.predict(X)[0]
        output = self.nn.forward(X)[0]

        sorted_probs = sorted(zip(class_names, output), key=lambda x: x[1], reverse=True)

        result = f"Распознанный класс: {class_names[prediction]}\n\n"
        for cls, prob in sorted_probs:
            result += f"{cls}: {prob:.4f}\n"

        # Создаем новое окно с крупным шрифтом
        top = tk.Toplevel(self.root)
        top.title("Результат распознавания")

        label = tk.Label(top, text=result, font=("Arial", 14), justify="left")
        label.pack(padx=20, pady=20)

        btn = ttk.Button(top, text="Закрыть", command=top.destroy)
        btn.pack(pady=(0, 20))

    def clear_drawn_dataset(self):
        """Очистка датасета нарисованных образов"""
        if messagebox.askyesno("Подтверждение",
                               "Удалить все нарисованные образы?"):
            if os.path.exists(self.drawn_dataset_file):
                os.remove(self.drawn_dataset_file)
            self.drawn_data = []
            self.update_info()
            messagebox.showinfo("Успех", "Датасет нарисованных образов очищен!")

    def clear_generated_dataset(self):
        """Очистка датасета сгенерированных образов"""
        if messagebox.askyesno("Подтверждение",
                               "Удалить все сгенерированные образы?"):
            if os.path.exists(self.generated_dataset_file):
                os.remove(self.generated_dataset_file)
            self.generated_data = []
            self.update_info()
            messagebox.showinfo("Успех", "Датасет сгенерированных образов очищен!")

    def clear_all_datasets(self):
        """Очистка всех датасетов"""
        if messagebox.askyesno("Подтверждение",
                               "Удалить ВСЕ датасеты?"):
            if os.path.exists(self.drawn_dataset_file):
                os.remove(self.drawn_dataset_file)
            if os.path.exists(self.generated_dataset_file):
                os.remove(self.generated_dataset_file)
            self.drawn_data = []
            self.generated_data = []
            self.update_info()
            messagebox.showinfo("Успех", "Все датасеты очищены!")

    def update_info(self):
        """Обновление информации о датасетах"""
        self.load_datasets()
        info = f"Нарисовано: {len(self.drawn_data)}\n"
        info += f"Сгенерировано: {len(self.generated_data)}\n"
        info += f"Всего: {len(self.drawn_data) + len(self.generated_data)}"
        self.info_label.config(text=info)


if __name__ == "__main__":
    root = tk.Tk()
    app = PatternRecognitionApp(root)
    root.mainloop()
