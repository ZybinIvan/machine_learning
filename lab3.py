import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import json
import os
import threading
from dataclasses import dataclass

# ----------------- Константы и классы -----------------

GRID_N = 16
CLASSES = ['+', '-', '/', '*', '√', '%']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
DATASET_PATH = "dataset.json"  # файл датасета по умолчанию

CELL = 22  # размеры визуальной клетки на Canvas

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

@dataclass
class MLPConfig:
    input_size: int = GRID_N * GRID_N
    hidden_size: int = 8
    output_size: int = len(CLASSES)
    lr: float = 0.1
    seed: int | None = 42

class MLP:
    def __init__(self, cfg: MLPConfig):
        if cfg.seed is not None:
            np.random.seed(cfg.seed)
        self.W1 = np.random.randn(cfg.input_size, cfg.hidden_size) * np.sqrt(2.0 / (cfg.input_size + cfg.hidden_size))
        self.b1 = np.zeros((1, cfg.hidden_size))
        self.W2 = np.random.randn(cfg.hidden_size, cfg.output_size) * np.sqrt(2.0 / (cfg.hidden_size + cfg.output_size))
        self.b2 = np.zeros((1, cfg.output_size))
        self.lr = cfg.lr

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = sigmoid(z2)
        return a1, a2

    def predict_proba(self, X):
        _, a2 = self.forward(X)
        return a2  # вероятности по каждому выходу (сигмоида)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1), proba

    def train(self, X, Y, epochs=500, lr=None, callback=None, stop_flag=None):
        if lr is not None:
            self.lr = lr
        N = X.shape[0]
        for ep in range(1, epochs + 1):
            if stop_flag and stop_flag.is_set():
                break
            a1, y_hat = self.forward(X)
            loss = np.mean((y_hat - Y) ** 2)

            delta2 = (y_hat - Y) * (y_hat * (1.0 - y_hat))
            grad_W2 = (a1.T @ delta2) / N
            grad_b2 = np.mean(delta2, axis=0, keepdims=True)

            delta1 = (delta2 @ self.W2.T) * (a1 * (1.0 - a1))
            grad_W1 = (X.T @ delta1) / N
            grad_b1 = np.mean(delta1, axis=0, keepdims=True)

            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1

            if callback:
                callback(ep, float(loss))

    def save_weights(self, path):
        weights = {
            "W1": self.W1.tolist(),
            "b1": self.b1.tolist(),
            "W2": self.W2.tolist(),
            "b2": self.b2.tolist(),
            "lr": self.lr,
            "input_size": GRID_N * GRID_N,
            "hidden_size": self.W1.shape[1],
            "output_size": self.W2.shape[1],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(weights, f, ensure_ascii=False, indent=2)

    def load_weights(self, path):
        with open(path, "r", encoding="utf-8") as f:
            w = json.load(f)
        assert w["input_size"] == GRID_N * GRID_N
        assert w["hidden_size"] == self.W1.shape[1]
        assert w["output_size"] == self.W2.shape[1]
        self.W1 = np.array(w["W1"], dtype=float)
        self.b1 = np.array(w["b1"], dtype=float)
        self.W2 = np.array(w["W2"], dtype=float)
        self.b2 = np.array(w["b2"], dtype=float)
        self.lr = float(w.get("lr", self.lr))

# ----------------- Вспомогательные генераторы символов -----------------

def empty_grid():
    return np.zeros((GRID_N, GRID_N), dtype=float)

def draw_hline(g, row, c0, c1):
    c0, c1 = max(0, c0), min(GRID_N-1, c1)
    g[row, c0:c1+1] = 1.0

def draw_vline(g, col, r0, r1):
    r0, r1 = max(0, r0), min(GRID_N-1, r1)
    g[r0:r1+1, col] = 1.0

def draw_diag(g, start, end):
    # Брезенхем (упрощённый) для тонкой линии
    r0, c0 = start
    r1, c1 = end
    dr = 1 if r1 >= r0 else -1
    dc = 1 if c1 >= c0 else -1
    r, c = r0, c0
    while True:
        if 0 <= r < GRID_N and 0 <= c < GRID_N:
            g[r, c] = 1.0
        if r == r1 and c == c1:
            break
        if abs((r+dr)-r0) * abs(c1-c0) < abs((c+dc)-c0) * abs(r1-r0):
            r += dr
        else:
            c += dc

def template_plus():
    g = empty_grid()
    mid = GRID_N // 2
    draw_hline(g, mid, 3, GRID_N-4)
    draw_vline(g, mid, 3, GRID_N-4)
    return g

def template_minus():
    g = empty_grid()
    mid = GRID_N // 2
    draw_hline(g, mid, 2, GRID_N-3)
    return g

def template_slash():
    g = empty_grid()
    # диагональ /
    draw_diag(g, (GRID_N-3, 2), (2, GRID_N-3))
    return g

def template_star():
    g = empty_grid()
    mid = GRID_N // 2
    draw_hline(g, mid, 3, GRID_N-4)
    draw_vline(g, mid, 3, GRID_N-4)
    draw_diag(g, (3, 3), (GRID_N-4, GRID_N-4))
    draw_diag(g, (GRID_N-4, 3), (3, GRID_N-4))
    return g

def template_sqrt():
    g = empty_grid()
    # галочка √
    draw_diag(g, (4, 6), (10, 9))
    draw_diag(g, (10, 9), (4, 13))
    return g

def template_percent():
    g = empty_grid()
    # диагональ %
    draw_diag(g, (GRID_N-3, 3), (3, GRID_N-4))
    # две точки (2x2)
    g[3:5, 3:5] = 1.0
    g[GRID_N-5:GRID_N-3, GRID_N-5:GRID_N-3] = 1.0
    return g

TEMPLATES = {
    '+': template_plus,
    '-': template_minus,
    '/': template_slash,
    '*': template_star,
    '√': template_sqrt,
    '%': template_percent,
}

def translate_grid(g, dx, dy):
    out = np.zeros_like(g)
    r0 = max(0, dy)
    r1 = min(GRID_N, GRID_N + dy)
    c0 = max(0, dx)
    c1 = min(GRID_N, GRID_N + dx)
    out[r0:r1, c0:c1] = g[r0 - dy:r1 - dy, c0 - dx:c1 - dx]
    return out

def thicken(g, iters=1):
    out = g.copy()
    for _ in range(iters):
        idx = np.argwhere(out > 0.5)
        for (r, c) in idx:
            for rr, cc in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
                if 0 <= rr < GRID_N and 0 <= cc < GRID_N:
                    out[rr, cc] = 1.0
    return out

def jitter_template(base):
    g = base.copy()
    # случайное смещение -1..1 по каждой оси
    dx = np.random.randint(-1, 2)
    dy = np.random.randint(-1, 2)
    g = translate_grid(g, dx, dy)
    # небольшое утолщение
    if np.random.rand() < 0.7:
        g = thicken(g, iters=np.random.randint(1, 3))
    # легкий шум включений рядом
    ones = np.argwhere(g > 0.5)
    for (r, c) in ones[np.random.choice(len(ones), size=max(1, len(ones)//10), replace=False)] if len(ones) else []:
        for rr, cc in [(r-1,c-1),(r-1,c+1),(r+1,c-1),(r+1,c+1)]:
            if 0 <= rr < GRID_N and 0 <= cc < GRID_N and np.random.rand() < 0.3:
                g[rr, cc] = 1.0
    return g

# ----------------- Приложение (Tkinter GUI) -----------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Распознавание образов — Арифметические операции")
        self.resizable(False, False)

        self.grid_data = np.zeros((GRID_N, GRID_N), dtype=float)
        self.drawing = False
        self.erasing = False
        self.current_label = CLASSES[0]

        self.samples_X: list[list[float]] = []
        self.samples_y: list[int] = []

        self.model = MLP(MLPConfig())

        # Параметры обучения
        self.lr_var = tk.StringVar(value="0.1")
        self.epochs_var = tk.StringVar(value="500")
        self.hidden_layers_var = tk.StringVar(value="1")  # должно быть 1

        # Параметры генерации
        self.gen_count_var = tk.StringVar(value="20")

        self._build_ui()
        self.stop_flag = threading.Event()

    def _build_ui(self):
        left = ttk.Frame(self)
        left.grid(row=0, column=0, padx=10, pady=10)

        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="ns", padx=10, pady=10)

        # Canvas 16x16
        canvas_size = GRID_N * CELL + 1
        self.canvas = tk.Canvas(left, width=canvas_size, height=canvas_size, bg="white", highlightthickness=1, highlightbackground="#888")
        self.canvas.grid(row=0, column=0)
        self._draw_grid()

        tools = ttk.Frame(left)
        tools.grid(row=1, column=0, pady=(8, 0), sticky="w")

        ttk.Button(tools, text="Очистить", command=self.clear_canvas).grid(row=0, column=0, padx=2)
        ttk.Button(tools, text="Инвертировать", command=self.invert_canvas).grid(row=0, column=1, padx=2)

        # Обработчики мыши
        self.canvas.bind("<Button-1>", self.on_lmb_down)
        self.canvas.bind("<B1-Motion>", self.on_lmb_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_lmb_up)

        self.canvas.bind("<Button-3>", self.on_rmb_down)
        self.canvas.bind("<B3-Motion>", self.on_rmb_move)
        self.canvas.bind("<ButtonRelease-3>", self.on_rmb_up)

        # Классы
        cls_box = ttk.LabelFrame(right, text="Класс")
        cls_box.grid(row=0, column=0, sticky="ew", pady=5)
        row, col = 0, 0
        for c in CLASSES:
            b = ttk.Button(cls_box, text=c, command=lambda cc=c: self.set_label(cc), width=4)
            b.grid(row=row, column=col, padx=2, pady=2)
            col += 1
            if col == 6:
                row += 1
                col = 0
        self.label_var = tk.StringVar(value=f"Текущий класс: {self.current_label}")
        ttk.Label(cls_box, textvariable=self.label_var).grid(row=2, column=0, columnspan=6, sticky="w", padx=4, pady=2)

        # Датасет
        ds_box = ttk.LabelFrame(right, text="Датасет")
        ds_box.grid(row=1, column=0, sticky="ew", pady=5)
        ttk.Button(ds_box, text="Добавить в датасет", command=self.add_sample).grid(row=0, column=0, padx=2, pady=2)
        ttk.Label(ds_box, text=f"Файл: {DATASET_PATH}").grid(row=0, column=1, columnspan=2, sticky="w", padx=6)

        ttk.Label(ds_box, text="Сгенерировать (на класс):").grid(row=1, column=0, sticky="w")
        ttk.Entry(ds_box, textvariable=self.gen_count_var, width=8).grid(row=1, column=1, sticky="w")
        ttk.Button(ds_box, text="Сгенерировать датасет", command=self.generate_dataset).grid(row=1, column=2, padx=2, pady=2)

        self.ds_count_var = tk.StringVar(value="Примеров (в памяти): 0")
        ttk.Label(ds_box, textvariable=self.ds_count_var).grid(row=2, column=0, columnspan=3, sticky="w", padx=4, pady=2)

        # Обучение
        train_box = ttk.LabelFrame(right, text="Обучение")
        train_box.grid(row=2, column=0, sticky="ew", pady=5)
        ttk.Label(train_box, text="Норма обучения:").grid(row=0, column=0, sticky="w")
        ttk.Entry(train_box, textvariable=self.lr_var, width=8).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(train_box, text="Эпохи:").grid(row=0, column=2, sticky="w")
        ttk.Entry(train_box, textvariable=self.epochs_var, width=8).grid(row=0, column=3, sticky="w", padx=4)
        ttk.Label(train_box, text="Скрытых слоёв (1):").grid(row=1, column=0, columnspan=1, sticky="w", pady=2)
        ttk.Entry(train_box, textvariable=self.hidden_layers_var, width=8).grid(row=1, column=1, sticky="w", padx=4)

        ttk.Button(train_box, text="Обучить сеть", command=self.start_training).grid(row=2, column=0, padx=2, pady=4, sticky="w")
        ttk.Button(train_box, text="Остановить", command=self.stop_training).grid(row=2, column=1, padx=2, pady=4, sticky="w")
        ttk.Button(train_box, text="Сохранить веса", command=self.save_weights).grid(row=2, column=2, padx=2, pady=4, sticky="w")
        ttk.Button(train_box, text="Загрузить веса", command=self.load_weights).grid(row=2, column=3, padx=2, pady=4, sticky="w")

        self.status_var = tk.StringVar(value="Статус: готово")
        ttk.Label(train_box, textvariable=self.status_var).grid(row=3, column=0, columnspan=4, sticky="w")

        # Распознавание
        infer_box = ttk.LabelFrame(right, text="Распознавание")
        infer_box.grid(row=3, column=0, sticky="ew", pady=5)
        ttk.Button(infer_box, text="Предсказать", command=self.predict_current).grid(row=0, column=0, padx=2, pady=4)
        self.pred_var = tk.StringVar(value="Предсказание: —")
        ttk.Label(infer_box, textvariable=self.pred_var).grid(row=0, column=1, padx=6, sticky="w")

        ttk.Label(infer_box, text="Вероятности:").grid(row=1, column=0, sticky="w")
        self.prob_box = tk.Text(infer_box, width=24, height=7)
        self.prob_box.grid(row=2, column=0, columnspan=2, padx=2, pady=2)
        self.prob_box.configure(state="disabled")

    # ---------- Рисование ----------
    def _draw_grid(self):
        self.canvas.delete("all")
        for i in range(GRID_N):
            for j in range(GRID_N):
                x0 = j * CELL + 1
                y0 = i * CELL + 1
                x1 = x0 + CELL - 2
                y1 = y0 + CELL - 2
                val = self.grid_data[i, j]
                fill = "black" if val > 0.5 else "white"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#ddd", tags=f"cell-{i}-{j}")

    def _update_cell(self, i, j):
        fill = "black" if self.grid_data[i, j] > 0.5 else "white"
        items = self.canvas.find_withtag(f"cell-{i}-{j}")
        if items:
            self.canvas.itemconfig(items[0], fill=fill)

    def set_cell(self, event, value):
        x, y = event.x, event.y
        j = x // CELL
        i = y // CELL
        if 0 <= i < GRID_N and 0 <= j < GRID_N:
            self.grid_data[i, j] = value
            self._update_cell(i, j)

    def on_lmb_down(self, event):
        self.drawing = True
        self.set_cell(event, 1.0)

    def on_lmb_move(self, event):
        if self.drawing:
            self.set_cell(event, 1.0)

    def on_lmb_up(self, event):
        self.drawing = False

    def on_rmb_down(self, event):
        self.erasing = True
        self.set_cell(event, 0.0)

    def on_rmb_move(self, event):
        if self.erasing:
            self.set_cell(event, 0.0)

    def on_rmb_up(self, event):
        self.erasing = False

    def clear_canvas(self):
        self.grid_data[:, :] = 0.0
        self._draw_grid()

    def invert_canvas(self):
        self.grid_data = 1.0 - self.grid_data
        self._draw_grid()

    def set_label(self, c):
        self.current_label = c
        self.label_var.set(f"Текущий класс: {self.current_label}")

    # ---------- Датасет ----------
    def grid_to_vector(self):
        return self.grid_data.reshape(-1).astype(float).tolist()

    def ensure_dataset_file(self):
        if not os.path.exists(DATASET_PATH):
            data = {"classes": CLASSES, "X": [], "y": []}
            with open(DATASET_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return data
        else:
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                return json.load(f)

    def append_to_dataset_file(self, x_vec, y_idx):
        data = self.ensure_dataset_file()
        # проверка совместимости классов
        if data.get("classes") != CLASSES:
            raise ValueError("Список классов в файле датасета несовместим")
        data["X"].append(x_vec)
        data["y"].append(int(y_idx))
        with open(DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_sample(self):
        x = self.grid_to_vector()
        y = CLASS_TO_IDX[self.current_label]
        self.samples_X.append(x)
        self.samples_y.append(y)
        try:
            self.append_to_dataset_file(x, y)
            messagebox.showinfo("Датасет", f"Добавлен пример класса '{self.current_label}' в {DATASET_PATH}")
        except Exception as e:
            messagebox.showerror("Датасет", f"Ошибка записи: {e}")
        self.ds_count_var.set(f"Примеров (в памяти): {len(self.samples_X)}")

    def generate_dataset(self):
        try:
            n = int(self.gen_count_var.get())
            if n <= 0:
                raise ValueError("n <= 0")
        except Exception:
            messagebox.showerror("Генерация", "Некорректное число на класс")
            return
        data = self.ensure_dataset_file()
        if data.get("classes") != CLASSES:
            messagebox.showerror("Датасет", "Список классов в файле датасета несовместим")
            return
        # генерируем
        for sym in CLASSES:
            base = TEMPLATES[sym]()
            for _ in range(n):
                g = jitter_template(base)
                x = g.reshape(-1).astype(float).tolist()
                y = CLASS_TO_IDX[sym]
                data["X"].append(x)
                data["y"].append(int(y))
                # также добавим в оперативную память
                self.samples_X.append(x)
                self.samples_y.append(y)
        with open(DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.ds_count_var.set(f"Примеров (в памяти): {len(self.samples_X)}")
        messagebox.showinfo("Генерация", f"Сгенерировано: {n*len(CLASSES)} примеров")

    # ---------- Обучение ----------
    def start_training(self):
        if self.hidden_layers_var.get().strip() != "1":
            messagebox.showerror("Параметры", "Число скрытых слоёв должно быть 1 по условию")
            return
        if not self.samples_X:
            messagebox.showwarning("Данные", "Датасет в памяти пуст")
            return
        try:
            lr = float(self.lr_var.get())
            epochs = int(self.epochs_var.get())
        except ValueError:
            messagebox.showerror("Параметры", "Неверные значения нормы обучения или эпох")
            return

        X = np.array(self.samples_X, dtype=float)
        y_idx = np.array(self.samples_y, dtype=int)
        Y = np.zeros((len(y_idx), len(CLASSES)), dtype=float)
        Y[np.arange(len(y_idx)), y_idx] = 1.0

        self.stop_flag.clear()
        def cb(ep, loss):
            self.status_var.set(f"Эпоха {ep}/{epochs} | Loss={loss:.6f}")
            self.update_idletasks()

        def run():
            self.status_var.set("Обучение...")
            self.model.train(X, Y, epochs=epochs, lr=lr, callback=cb, stop_flag=self.stop_flag)
            self.status_var.set("Остановлено" if self.stop_flag.is_set() else "Готово")

        threading.Thread(target=run, daemon=True).start()

    def stop_training(self):
        self.stop_flag.set()

    def save_weights(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            self.model.save_weights(path)
            messagebox.showinfo("Веса", f"Сохранено: {path}")
        except Exception as e:
            messagebox.showerror("Веса", f"Ошибка сохранения: {e}")

    def load_weights(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            self.model.load_weights(path)
            messagebox.showinfo("Веса", f"Загружено: {path}")
        except Exception as e:
            messagebox.showerror("Веса", f"Ошибка загрузки: {e}")

    # ---------- Предсказание ----------
    def predict_current(self):
        x = np.array(self.grid_to_vector(), dtype=float).reshape(1, -1)
        idx, proba = self.model.predict(x)
        idx = int(idx[0])
        probs = proba[0]
        sym = CLASSES[idx]
        self.pred_var.set(f"Предсказание: {sym} (p={probs[idx]:.3f})")
        # показать все вероятности
        lines = []
        for i, c in enumerate(CLASSES):
            lines.append(f"{c}: {probs[i]:.3f}")
        self.prob_box.configure(state="normal")
        self.prob_box.delete("1.0", "end")
        self.prob_box.insert("1.0", "\n".join(lines))
        self.prob_box.configure(state="disabled")

if __name__ == "__main__":
    App().mainloop()
