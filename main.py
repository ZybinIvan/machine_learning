import tkinter as tk
from tkinter import messagebox
import numpy as np


class HebbNeuron:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)
        self.bias = 0

    def predict(self, x):
        s = np.dot(self.weights, x) + self.bias
        return 1 if s > 0 else -1

    def train(self, X, Y):
        for x, y in zip(X, Y):
            self.weights += y * x
            self.bias += y


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Обучение нейрона Хебба')
        self.grid_size = 6
        self.cell_size = 36
        self.canvas_size = self.grid_size * self.cell_size

        self.canvas = tk.Canvas(self, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=3)

        self.cells = []
        for r in range(self.grid_size):
            row = []
            for c in range(self.grid_size):
                x1 = c * self.cell_size
                y1 = r * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                cell = self.canvas.create_rectangle(x1, y1, x2, y2, fill='white', outline='gray')
                row.append(cell)
            self.cells.append(row)

        self.canvas.bind('<Button-1>', self.paint_cell_start)
        self.canvas.bind('<B1-Motion>', self.paint_cell_motion)

        tk.Label(self, text="Обучаемая буква:").grid(row=1, column=0, padx=5, pady=5)
        self.training_letter_entry = tk.Entry(self, width=8)
        self.training_letter_entry.grid(row=1, column=1, padx=5, pady=5)

        self.add_pattern_button = tk.Button(self, text="Добавить образ", command=self.add_pattern)
        self.add_pattern_button.grid(row=1, column=2, padx=5, pady=5)

        self.recognize_button = tk.Button(self, text="Распознать", command=self.recognize_pattern)
        self.recognize_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

        self.clear_button = tk.Button(self, text="Очистить", command=self.clear_canvas)
        self.clear_button.grid(row=2, column=2, padx=5, pady=5)

        self.neuron = HebbNeuron(self.grid_size * self.grid_size)
        self.letter_classes = {}  # {'A': +1, 'B': -1}
        self.data = []
        self.targets = []

    def paint_cell(self, event):
        c = event.x // self.cell_size
        r = event.y // self.cell_size
        if 0 <= c < self.grid_size and 0 <= r < self.grid_size:
            self.canvas.itemconfig(self.cells[r][c], fill='black')

    def paint_cell_start(self, event):
        self.paint_cell(event)

    def paint_cell_motion(self, event):
        self.paint_cell(event)

    def get_canvas_pattern(self):
        pattern = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                color = self.canvas.itemcget(self.cells[r][c], 'fill')
                pattern.append(1 if color == 'black' else -1)
        return np.array(pattern)

    def add_pattern(self):
        letter = self.training_letter_entry.get().strip()
        if not letter:
            messagebox.showwarning("Ошибка", "Введите букву для обучения.")
            return

        # Определяем класс для буквы
        if letter not in self.letter_classes:
            if len(self.letter_classes) == 0:
                self.letter_classes[letter] = 1
                messagebox.showinfo("Инфо", f'Буква "{letter}" назначена классу +1.')
            elif len(self.letter_classes) == 1:
                self.letter_classes[letter] = -1
                messagebox.showinfo("Инфо", f'Буква "{letter}" назначена классу -1.')
            else:
                messagebox.showwarning("Ошибка", "Можно обучать только две разные буквы.")
                return

        label = self.letter_classes[letter]
        pattern = self.get_canvas_pattern()

        self.data.append(pattern)
        self.targets.append(label)
        self.neuron.train([pattern], [label])
        messagebox.showinfo("Ок", f'Образ для буквы "{letter}" добавлен (класс {label}).')

    def recognize_pattern(self):
        if len(self.letter_classes) < 2:
            messagebox.showwarning("Ошибка", "Сначала обучите две буквы.")
            return

        pattern = self.get_canvas_pattern()
        result = self.neuron.predict(pattern)

        for letter, label in self.letter_classes.items():
            if label == result:
                messagebox.showinfo("Результат", f'Образ распознан как буква: "{letter}"')
                return

        messagebox.showinfo("Результат", "Образ не распознан.")

    def clear_canvas(self):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                self.canvas.itemconfig(self.cells[r][c], fill='white')


if __name__ == '__main__':
    app = App()
    app.mainloop()
