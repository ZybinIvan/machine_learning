import tkinter as tk
import numpy as np

class HebbNeuron:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)
        self.bias = 0

    def predict(self, x):
        s = np.dot(self.weights, x) + self.bias
        return 1 if s >= 0 else -1

    def train(self, X, Y,):
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
        self.canvas.grid(row=0, column=0, columnspan=4)

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

        tk.Label(self, text="Букву 1 (класс +1):").grid(row=1, column=0, padx=5, pady=5)
        self.letter1_entry = tk.Entry(self, width=5)
        self.letter1_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self, text="Букву 2 (класс -1):").grid(row=2, column=0, padx=5, pady=5)
        self.letter2_entry = tk.Entry(self, width=5)
        self.letter2_entry.grid(row=2, column=1, padx=5, pady=5)

        self.train_button_1 = tk.Button(self, text="Обучить букве 1 (+1)", command=self.train_neuron1)
        self.train_button_1.grid(row=1, column=2, padx=5, pady=5)
        self.train_button_2 = tk.Button(self, text="Обучить букве 2 (-1)", command=self.train_neuron2)
        self.train_button_2.grid(row=2, column=2, padx=5, pady=5)

        self.recognize_button = tk.Button(self, text="Распознать", command=self.recognize_pattern)
        self.recognize_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

        self.clear_button = tk.Button(self, text="Очистить", command=self.clear_canvas)
        self.clear_button.grid(row=3, column=2, columnspan=2, sticky='we', padx=5, pady=5)

        self.neuron = HebbNeuron(self.grid_size * self.grid_size)
        self.letter1 = ""
        self.letter2 = ""

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

    def train_neuron1(self):
        self.letter1 = self.letter1_entry.get().strip()
        if not self.letter1:
            print("Введите первую букву.")
            return
        pattern = self.get_canvas_pattern()
        self.data.append(pattern)
        self.targets.append(1)  # +1 для первой буквы
        self.neuron.train([pattern], [1])
        print(f'Образ для буквы "{self.letter1}" добавлен в обучение (класс +1).')

    def train_neuron2(self):
        self.letter2 = self.letter2_entry.get().strip()
        if not self.letter2:
            print("Введите вторую букву.")
            return
        pattern = self.get_canvas_pattern()
        self.data.append(pattern)
        self.targets.append(-1)  # -1 для второй буквы
        self.neuron.train([pattern], [-1])
        print(f'Образ для буквы "{self.letter2}" добавлен в обучение (класс -1).')

    def recognize_pattern(self):
        pattern = self.get_canvas_pattern()
        result = self.neuron.predict(pattern)
        if result == 1:
            print(f'Образ распознан как буква: "{self.letter1}"')
        else:
            print(f'Образ распознан как буква: "{self.letter2}"')

    def clear_canvas(self):
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                self.canvas.itemconfig(self.cells[r][c], fill='white')

if __name__ == '__main__':
    app = App()
    app.mainloop()
