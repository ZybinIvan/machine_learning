import tkinter as tk
from tkinter import ttk, messagebox
import random
import math

# === для графика ===
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

LIGHT = "#ffffff"      # белые клетки
DARK = "#b58863"       # коричневые клетки
OK_COLOR = "#2ecc71"   # зелёный
BAD_COLOR = "#e74c3c"  # красный
CANVAS_SIZE = 640

def random_state(n: int):
    return [random.randrange(n) for _ in range(n)]

def cost(state):
    n = len(state)
    c = 0
    for i in range(n):
        ri = state[i]
        for j in range(i + 1, n):
            rj = state[j]
            if ri == rj or abs(ri - rj) == abs(i - j):
                c += 1
    return c

def conflicts_per_queen(state):
    n = len(state)
    conf = [0] * n
    for i in range(n):
        ri = state[i]
        for j in range(i + 1, n):
            rj = state[j]
            if ri == rj or abs(ri - rj) == abs(i - j):
                conf[i] += 1
                conf[j] += 1
    return conf

def neighbor(state):
    n = len(state)
    s = state[:]
    if n <= 1:
        return s
    if random.random() < 0.5:
        col = random.randrange(n)
        new_row = random.randrange(n)
        if n > 1:
            while new_row == s[col]:
                new_row = random.randrange(n)
        s[col] = new_row
    else:
        i = random.randrange(n)
        j = random.randrange(n)
        while j == i:
            j = random.randrange(n)
        s[i], s[j] = s[j], s[i]
    return s

def simulated_annealing(n, T0=10.0, alpha=0.995, max_steps=30000, restarts=3):
    best_state = None
    best_cost = math.inf
    history = []  # история лучшей стоимости во времени (монотонная)

    for _ in range(max(1, int(restarts))):
        curr = random_state(n)
        curr_cost = cost(curr)
        T = float(T0)

        best_local = curr_cost
        steps = 0
        while steps < max_steps and curr_cost > 0 and T > 1e-12:
            nb = neighbor(curr)
            nb_cost = cost(nb)
            delta = nb_cost - curr_cost
            if delta <= 0:
                curr, curr_cost = nb, nb_cost
            else:
                if random.random() < math.exp(-delta / max(T, 1e-12)):
                    curr, curr_cost = nb, nb_cost

            if curr_cost < best_local:
                best_local = curr_cost
            history.append(best_local)

            T *= alpha
            steps += 1

        history.append(best_local)

        if best_local < best_cost:
            best_state, best_cost = curr, best_local
        if best_cost == 0:
            break

    return best_state, best_cost, history

class AnnealingQueensGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("N-ферзей — имитация отжига (Tkinter)")
        self.state = None

        # Верхняя панель с параметрами
        self.controls = ttk.Frame(root, padding=10)
        self.controls.pack(side=tk.TOP, fill=tk.X)

        self.n_var = tk.StringVar(value="8")
        self.t0_var = tk.StringVar(value="10.0")
        self.alpha_var = tk.StringVar(value="0.995")
        self.steps_var = tk.StringVar(value="30000")
        self.restarts_var = tk.StringVar(value="3")

        self._add_labeled_entry("N:", self.n_var, 0, 0, width=8)
        self._add_labeled_entry("T0:", self.t0_var, 0, 2, width=8)
        self._add_labeled_entry("alpha:", self.alpha_var, 0, 4, width=8)
        self._add_labeled_entry("steps:", self.steps_var, 0, 6, width=10)
        self._add_labeled_entry("restarts:", self.restarts_var, 0, 8, width=8)

        self.run_btn = ttk.Button(self.controls, text="Старт", command=self.run)
        self.run_btn.grid(row=0, column=10, padx=8)

        # Центральная область: слева доска, справа график
        self.content = ttk.Frame(root)
        self.content.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.left_frame = ttk.Frame(self.content)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10,5), pady=10)

        self.right_frame = ttk.Frame(self.content)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5,10), pady=10)

        # Доска
        self.canvas = tk.Canvas(self.left_frame, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="#fafafa", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

        # График (Matplotlib внутри правого фрейма)
        self.fig = None
        self.ax = None
        self.fig_canvas = None
        if MATPLOTLIB_OK:
            self.fig = Figure(figsize=(6.5, 4.0), dpi=100)
            self.ax = self.fig.add_subplot(111)
            self.ax.set_title("Имитация отжига: сходимость")
            self.ax.set_xlabel("Итерация")
            self.ax.set_ylabel("Стоимость (число конфликтов)")
            self.ax.grid(True, linestyle="--", alpha=0.5)
            self.fig_canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
            self.fig_canvas.draw()
            self.fig_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        else:
            ttk.Label(
                self.right_frame,
                text="Matplotlib не найден. Установите пакет для показа графика."
            ).pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.status = ttk.Label(root, text="Готово")
        self.status.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0,10))

    def _add_labeled_entry(self, text, var, r, c, width=10):
        lab = ttk.Label(self.controls, text=text)
        lab.grid(row=r, column=c, sticky=tk.W, padx=(0,4))
        ent = ttk.Entry(self.controls, textvariable=var, width=width)
        ent.grid(row=r, column=c+1, padx=(0,10))
        return ent

    def run(self):
        try:
            n = int(self.n_var.get())
            T0 = float(self.t0_var.get())
            alpha = float(self.alpha_var.get())
            steps = int(self.steps_var.get())
            restarts = int(self.restarts_var.get())
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте, что N, T0, alpha, steps и restarts заданы корректно.")
            return

        if n <= 0:
            messagebox.showerror("Ошибка", "N должно быть положительным.")
            return
        if not (0.0 < alpha < 1.0):
            messagebox.showerror("Ошибка", "alpha должно быть в (0, 1).")
            return
        if T0 <= 0.0:
            messagebox.showerror("Ошибка", "T0 должно быть положительным.")
            return

        self.status.config(text="Вычисление...")
        self.root.update_idletasks()

        state, c, history = simulated_annealing(
            n, T0=T0, alpha=alpha, max_steps=steps, restarts=restarts
        )
        self.state = state
        self.draw_board()
        if c == 0:
            self.status.config(text=f"Решение найдено: стоимость=0, N={n}")
        else:
            self.status.config(text=f"Локально лучшее: стоимость={c}, N={n}")

        self.update_convergence(history)

    def draw_board(self):
        self.canvas.delete("all")
        if not self.state:
            return
        n = len(self.state)
        cell = max(6, CANVAS_SIZE // max(1, n))
        board_px = cell * n
        ox = (CANVAS_SIZE - board_px) // 2
        oy = (CANVAS_SIZE - board_px) // 2

        for r in range(n):
            for c in range(n):
                x0 = ox + c * cell
                y0 = oy + r * cell
                x1 = x0 + cell
                y1 = y0 + cell
                fill = LIGHT if (r + c) % 2 == 0 else DARK
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline=fill)

        perq = conflicts_per_queen(self.state)
        rad = cell * 0.35
        for col, row in enumerate(self.state):
            cx = ox + col * cell + cell / 2
            cy = oy + row * cell + cell / 2
            color = OK_COLOR if perq[col] == 0 else BAD_COLOR
            self.canvas.create_oval(cx - rad, cy - rad, cx + rad, cy + rad,
                                    fill=color, outline="#222222", width=2)

    def update_convergence(self, history):
        if not MATPLOTLIB_OK or self.ax is None or self.fig_canvas is None:
            messagebox.showwarning(
                "Внимание",
                "Для графика требуется matplotlib. Установите пакет и повторите."
            )
            return

        # Подвыборка для производительности на длинной истории
        data = history
        max_points = 5000
        if len(data) > max_points:
            step = max(1, len(data) // max_points)
            data = data[::step]

        self.ax.clear()
        self.ax.plot(range(1, len(data) + 1), data, color="#1f77b4", linewidth=1.25)
        self.ax.set_xlabel("Итерация")
        self.ax.set_ylabel("Стоимость (число конфликтов)")
        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.ax.set_title("Имитация отжига: сходимость")
        self.fig_canvas.draw()

def main():
    root = tk.Tk()
    try:
        from tkinter import ttk  # noqa
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    app = AnnealingQueensGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
