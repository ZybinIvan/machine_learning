"""
Муравьиный алгоритм для TSP с ВИЗУАЛИЗАЦИЕЙ: отдельная генерация городов, затем запуск поиска.
Python 3.10+ / 3.14, Tkinter + Matplotlib
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
from typing import List, Tuple


class AntAlgorithmTSP:
    def __init__(self, cities: np.ndarray, num_ants: int,
                 alpha: float = 1.0, beta: float = 1.0,
                 rho: float = 0.5, Q: float = 1.0, iterations: int = 100):
        self.cities = cities
        self.num_cities = len(cities)
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.iterations = iterations

        self.distance_matrix = self._calculate_distances()
        self.pheromone = np.ones((self.num_cities, self.num_cities))

        self.best_path_history = []
        self.best_length_history = []
        self.avg_length_history = []
        self.best_path = None
        self.best_length = float('inf')

        self.current_iteration = 0
        self.current_ant_paths: List[Tuple[List[int], float]] = []

    def _calculate_distances(self) -> np.ndarray:
        n = self.num_cities
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[i][j] = np.linalg.norm(self.cities[i] - self.cities[j])
        return dist

    def _calculate_heuristic(self) -> np.ndarray:
        h = np.zeros_like(self.distance_matrix, dtype=float)
        nz = self.distance_matrix > 0
        h[nz] = 1.0 / self.distance_matrix[nz]
        return h

    def _calculate_probabilities(self, current_city: int, tabu_list: List[int]) -> np.ndarray:
        h = self._calculate_heuristic()
        probabilities = np.zeros(self.num_cities)
        unvisited = [j for j in range(self.num_cities) if j not in tabu_list]
        if not unvisited:
            return probabilities
        tau = self.pheromone[current_city, unvisited] ** self.alpha
        eta = h[current_city, unvisited] ** self.beta
        weights = tau * eta
        s = weights.sum()
        if s <= 0:
            probabilities[unvisited] = 1.0 / len(unvisited)
        else:
            probabilities[unvisited] = weights / s
        return probabilities

    def _select_next_city(self, current_city: int, tabu_list: List[int]) -> int:
        probabilities = self._calculate_probabilities(current_city, tabu_list)
        return np.random.choice(self.num_cities, p=probabilities)

    def _calculate_path_length(self, path: List[int]) -> float:
        length = 0.0
        for i in range(len(path) - 1):
            length += self.distance_matrix[path[i]][path[i + 1]]
        length += self.distance_matrix[path[-1]][path[0]]
        return length

    def _build_ant_path(self, start_city: int) -> Tuple[List[int], float]:
        tabu_list = [start_city]
        current_city = start_city
        while len(tabu_list) < self.num_cities:
            next_city = self._select_next_city(current_city, tabu_list)
            tabu_list.append(next_city)
            current_city = next_city
        return tabu_list, self._calculate_path_length(tabu_list)

    def _update_pheromone(self, ants_paths: List[Tuple[List[int], float]]):
        self.pheromone *= (1 - self.rho)
        for path, L in ants_paths:
            if L <= 0:
                continue
            delta = self.Q / L
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                self.pheromone[a][b] += delta
                self.pheromone[b][a] += delta
            a, b = path[-1], path[0]
            self.pheromone[a][b] += delta
            self.pheromone[b][a] += delta

    def run(self):
        self.best_path_history.clear()
        self.best_length_history.clear()
        self.avg_length_history.clear()

        for it in range(self.iterations):
            self.current_iteration = it
            ants_paths: List[Tuple[List[int], float]] = []
            for ant in range(self.num_ants):
                start_city = ant % self.num_cities
                path, L = self._build_ant_path(start_city)
                ants_paths.append((path, L))
                if L < self.best_length:
                    self.best_length = L
                    self.best_path = path

            self.current_ant_paths = ants_paths
            self._update_pheromone(ants_paths)

            lengths = [L for _, L in ants_paths]
            self.best_length_history.append(self.best_length)
            self.avg_length_history.append(float(np.mean(lengths)))
            self.best_path_history.append(self.best_path.copy())

        return self.best_path, self.best_length


class AntVisualizationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Муравьиный алгоритм — Визуализация")
        self.root.geometry("1400x900")

        self.algorithm: AntAlgorithmTSP | None = None
        self.is_running = False
        self.cities: np.ndarray | None = None

        self._create_widgets()

    def _create_widgets(self):
        left = ttk.Frame(self.root, width=280)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        left.pack_propagate(False)

        ttk.Label(left, text="Параметры", font=("Arial", 12, "bold")).pack()

        ttk.Label(left, text="Города:").pack(anchor=tk.W)
        self.num_cities_var = tk.IntVar(value=12)
        ttk.Spinbox(left, from_=3, to=50, textvariable=self.num_cities_var, width=12).pack(anchor=tk.W, pady=4)

        ttk.Label(left, text="Муравьи:").pack(anchor=tk.W)
        self.num_ants_var = tk.IntVar(value=20)
        ttk.Spinbox(left, from_=1, to=200, textvariable=self.num_ants_var, width=12).pack(anchor=tk.W, pady=4)

        ttk.Label(left, text="Итерации:").pack(anchor=tk.W)
        self.iterations_var = tk.IntVar(value=100)
        ttk.Spinbox(left, from_=1, to=2000, textvariable=self.iterations_var, width=12).pack(anchor=tk.W, pady=4)

        ttk.Label(left, text="Alpha:").pack(anchor=tk.W)
        self.alpha_var = tk.DoubleVar(value=1.0)
        ttk.Scale(left, from_=0.1, to=4.0, variable=self.alpha_var, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        self.alpha_label = ttk.Label(left, text="1.00"); self.alpha_label.pack(anchor=tk.W)
        self.alpha_var.trace("w", lambda *a: self.alpha_label.config(text=f"{self.alpha_var.get():.2f}"))

        ttk.Label(left, text="Beta:").pack(anchor=tk.W)
        self.beta_var = tk.DoubleVar(value=1.0)
        ttk.Scale(left, from_=0.1, to=4.0, variable=self.beta_var, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        self.beta_label = ttk.Label(left, text="1.00"); self.beta_label.pack(anchor=tk.W)
        self.beta_var.trace("w", lambda *a: self.beta_label.config(text=f"{self.beta_var.get():.2f}"))

        ttk.Label(left, text="Rho:").pack(anchor=tk.W)
        self.rho_var = tk.DoubleVar(value=0.5)
        ttk.Scale(left, from_=0.01, to=0.99, variable=self.rho_var, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        self.rho_label = ttk.Label(left, text="0.50"); self.rho_label.pack(anchor=tk.W)
        self.rho_var.trace("w", lambda *a: self.rho_label.config(text=f"{self.rho_var.get():.2f}"))

        # Новая кнопка: сгенерировать города (предпросмотр без запуска)
        ttk.Button(left, text="Сгенерировать города", command=self._on_generate_cities).pack(fill=tk.X, pady=10)

        # Запуск алгоритма по уже сгенерированным городам
        ttk.Button(left, text="Запустить", command=self._run_algorithm).pack(fill=tk.X, pady=6)

        ttk.Label(left, text="Результаты", font=("Arial", 10, "bold")).pack(pady=8)
        ttk.Label(left, text="Расстояние:").pack(anchor=tk.W)
        self.distance_var = tk.StringVar(value="-")
        ttk.Label(left, textvariable=self.distance_var, font=("Arial", 11, "bold")).pack(anchor=tk.W)

        ttk.Label(left, text="Итерация:").pack(anchor=tk.W)
        self.iteration_var = tk.StringVar(value="-")
        ttk.Label(left, textvariable=self.iteration_var).pack(anchor=tk.W)

        self.status_var = tk.StringVar(value="Готово")
        ttk.Label(left, textvariable=self.status_var, foreground="blue").pack(pady=10)

        right = ttk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.nb = ttk.Notebook(right); self.nb.pack(fill=tk.BOTH, expand=True)
        self.tab_paths = ttk.Frame(self.nb); self.nb.add(self.tab_paths, text="Города и пути")
        self.tab_convergence = ttk.Frame(self.nb); self.nb.add(self.tab_convergence, text="Сходимость")
        self.tab_pheromone = ttk.Frame(self.nb); self.nb.add(self.tab_pheromone, text="Феромоны")

        self.canvas_frame1 = ttk.Frame(self.tab_paths); self.canvas_frame1.pack(fill=tk.BOTH, expand=True)
        self.canvas_frame2 = ttk.Frame(self.tab_convergence); self.canvas_frame2.pack(fill=tk.BOTH, expand=True)
        self.canvas_frame3 = ttk.Frame(self.tab_pheromone); self.canvas_frame3.pack(fill=tk.BOTH, expand=True)

    def _generate_cities(self, n: int) -> np.ndarray:
        return np.random.rand(n, 2) * 100.0

    # Новое: обработчик генерации с предпросмотром точек
    def _on_generate_cities(self):
        self.cities = self._generate_cities(self.num_cities_var.get())
        self.distance_var.set("-")
        self.iteration_var.set("-")
        self._draw_cities_preview()
        self.status_var.set("Города сгенерированы, можно запускать поиск")

    # Новое: чистая отрисовка только точек городов (без маршрутов)
    def _draw_cities_preview(self):
        for w in self.canvas_frame1.winfo_children():
            w.destroy()
        fig = Figure(figsize=(7, 7), dpi=100)
        ax = fig.add_subplot(111)
        ax.scatter(self.cities[:, 0], self.cities[:, 1], c="red", s=90, zorder=4, edgecolors="black")
        for i, (x, y) in enumerate(self.cities):
            ax.annotate(str(i), (x, y), fontsize=9, ha="center", va="center", color="white", fontweight="bold")
        ax.set_title("Сгенерированные города (предпросмотр)")
        ax.set_xlabel("X"); ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=self.canvas_frame1).get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Очищаем другие вкладки до запуска
        for w in self.canvas_frame2.winfo_children():
            w.destroy()
        for w in self.canvas_frame3.winfo_children():
            w.destroy()

    def _run_algorithm(self):
        if self.is_running:
            messagebox.showwarning("Внимание", "Алгоритм уже запущен")
            return
        if self.cities is None:
            messagebox.showinfo("Требуется генерация", "Сначала нажмите «Сгенерировать города», затем «Запустить».")
            return
        self.is_running = True
        self.status_var.set("Выполнение...")
        threading.Thread(target=self._execute_algorithm, daemon=True).start()

    def _execute_algorithm(self):
        try:
            self.algorithm = AntAlgorithmTSP(
                cities=self.cities,
                num_ants=self.num_ants_var.get(),
                alpha=self.alpha_var.get(),
                beta=self.beta_var.get(),
                rho=self.rho_var.get(),
                Q=1.0,
                iterations=self.iterations_var.get(),
            )
            best_path, best_length = self.algorithm.run()
            self.root.after(0, self._update_visualization, best_path, best_length)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка", str(e)))
        finally:
            self.is_running = False

    def _update_visualization(self, best_path: List[int], best_length: float):
        self.distance_var.set(f"{best_length:.2f}")
        self.iteration_var.set(str(self.algorithm.current_iteration))

        for f in (self.canvas_frame1, self.canvas_frame2, self.canvas_frame3):
            for w in f.winfo_children():
                w.destroy()

        # Tab 1: Cities + paths
        fig1 = Figure(figsize=(7, 7), dpi=100); ax1 = fig1.add_subplot(111)
        ax1.scatter(self.cities[:, 0], self.cities[:, 1], c="red", s=90, zorder=4, edgecolors="black")
        for i, (x, y) in enumerate(self.cities):
            ax1.annotate(str(i), (x, y), fontsize=9, ha="center", va="center", color="white", fontweight="bold")
        for (path, _L) in self.algorithm.current_ant_paths[:5]:
            cyc = path + [path[0]]; pts = self.cities[cyc]
            ax1.plot(pts[:, 0], pts[:, 1], color="gray", alpha=0.25, linewidth=0.8, zorder=1)
        cyc_best = best_path + [best_path[0]]; pts_best = self.cities[cyc_best]
        ax1.plot(pts_best[:, 0], pts_best[:, 1], "b-", linewidth=2.2, zorder=3, label=f"Лучший маршрут: {best_length:.2f}")
        ax1.set_title("Города и найденный маршрут"); ax1.set_xlabel("X"); ax1.set_ylabel("Y")
        ax1.grid(True, alpha=0.3); ax1.set_aspect("equal", adjustable="box"); ax1.legend(loc="best")
        fig1.tight_layout()
        FigureCanvasTkAgg(fig1, master=self.canvas_frame1).get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tab 2: Convergence
        fig2 = Figure(figsize=(7, 7), dpi=100)
        ax2_1 = fig2.add_subplot(2, 2, 1)
        ax2_1.plot(self.algorithm.best_length_history, "b-", linewidth=2)
        ax2_1.set_title("Сходимость лучшего решения"); ax2_1.set_xlabel("Итерация"); ax2_1.set_ylabel("Расстояние"); ax2_1.grid(True, alpha=0.3)
        ax2_2 = fig2.add_subplot(2, 2, 2)
        ax2_2.plot(self.algorithm.avg_length_history, "r-", linewidth=2)
        ax2_2.set_title("Эволюция среднего решения"); ax2_2.set_xlabel("Итерация"); ax2_2.set_ylabel("Расстояние"); ax2_2.grid(True, alpha=0.3)
        ax2_3 = fig2.add_subplot(2, 2, 3)
        ax2_3.plot(self.algorithm.best_length_history, "b-", label="Лучшее", linewidth=2)
        ax2_3.plot(self.algorithm.avg_length_history, "r--", label="Среднее", linewidth=2)
        ax2_3.set_title("Сравнение"); ax2_3.set_xlabel("Итерация"); ax2_3.set_ylabel("Расстояние"); ax2_3.legend(); ax2_3.grid(True, alpha=0.3)
        ax2_4 = fig2.add_subplot(2, 2, 4)
        improvements = np.diff(self.algorithm.best_length_history); improvements = np.insert(improvements, 0, 0.0)
        ax2_4.bar(range(len(improvements)), improvements, color="g", alpha=0.7)
        ax2_4.set_title("Изменение лучшего"); ax2_4.set_xlabel("Итерация"); ax2_4.set_ylabel("Δ расстояния"); ax2_4.grid(True, alpha=0.3)
        fig2.tight_layout()
        FigureCanvasTkAgg(fig2, master=self.canvas_frame2).get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Tab 3: Pheromone heatmap (log scale)
        fig3 = Figure(figsize=(7, 7), dpi=100); ax3 = fig3.add_subplot(111)
        pheromone_log = np.log10(self.algorithm.pheromone + 1.0)
        im = ax3.imshow(pheromone_log, cmap="YlOrRd", aspect="auto")
        ax3.set_title("Распределение феромонов (log10)"); ax3.set_xlabel("Город j"); ax3.set_ylabel("Город i")
        cbar = fig3.colorbar(im, ax=ax3); cbar.set_label("log10(феромон)")
        fig3.tight_layout()
        FigureCanvasTkAgg(fig3, master=self.canvas_frame3).get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status_var.set("Завершено!")


if __name__ == "__main__":
    root = tk.Tk()
    app = AntVisualizationApp(root)
    root.mainloop()
