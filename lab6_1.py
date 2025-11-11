"""
Муравьиный алгоритм для TSP с ВИЗУАЛИЗАЦИЕЙ.
— Отдельная генерация городов, затем запуск поиска.
— Исследование сходимости по сетке параметров: таблица + построение графиков по таблице.
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

        self.best_path_history: List[List[int]] = []
        self.best_length_history: List[float] = []
        self.avg_length_history: List[float] = []
        self.best_path: List[int] | None = None
        self.best_length: float = float('inf')

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
        return int(np.random.choice(self.num_cities, p=probabilities))

    def _calculate_path_length(self, path: List[int]) -> float:
        length = 0.0
        for i in range(len(path) - 1):
            length += self.distance_matrix[path[i]][path[i + 1]]
        length += self.distance_matrix[path[-1]][path[0]]
        return float(length)

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

    def run(self) -> Tuple[List[int], float]:
        self.best_length_history.clear()
        self.avg_length_history.clear()
        self.best_path_history.clear()

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
        self.is_converging = False
        self.cities: np.ndarray | None = None

        self.convergence_results: List[List[float]] | None = None
        self.convergence_tree = None

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

        ttk.Button(left, text="Сгенерировать города", command=self._on_generate_cities).pack(fill=tk.X, pady=10)
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

        right = ttk.Frame(self.root); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.nb = ttk.Notebook(right); self.nb.pack(fill=tk.BOTH, expand=True)

        # Вкладка «Города и пути»
        self.tab_paths = ttk.Frame(self.nb); self.nb.add(self.tab_paths, text="Города и пути")
        self.canvas_frame1 = ttk.Frame(self.tab_paths); self.canvas_frame1.pack(fill=tk.BOTH, expand=True)

        # Вкладка «Сходимость»: диапазоны + кнопки + таблица
        self.tab_convergence = ttk.Frame(self.nb); self.nb.add(self.tab_convergence, text="Сходимость")

        panel = ttk.Frame(self.tab_convergence); panel.pack(fill=tk.X, padx=8, pady=6)

        def _mk_row(parent, label_txt, default, width=22):
            f = ttk.Frame(parent); f.pack(side=tk.LEFT, padx=6)
            ttk.Label(f, text=label_txt).pack(anchor=tk.W)
            e = ttk.Entry(f, width=width); e.insert(0, default); e.pack(anchor=tk.W)
            return e

        self.study_ants_entry = _mk_row(panel, "Муравьи (int):", "30")
        self.study_alpha_entry = _mk_row(panel, "Alpha:", "0.5,1.0,1.5,2.0")
        self.study_beta_entry  = _mk_row(panel, "Beta:",  "1.0,2.0,3.0")
        self.study_rho_entry   = _mk_row(panel, "Evaporation (ρ):", "0.3,0.6")
        self.study_q_entry     = _mk_row(panel, "Q (int):", "100,200")

        ctrl = ttk.Frame(self.tab_convergence); ctrl.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(ctrl, text="Исследовать на сходимость", command=self._run_convergence_study).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Построить графики", command=self._show_convergence_plots).pack(side=tk.LEFT, padx=8)

        self.convergence_frame = ttk.Frame(self.tab_convergence)
        self.convergence_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Вкладка «Феромоны»
        self.tab_pheromone = ttk.Frame(self.nb); self.nb.add(self.tab_pheromone, text="Феромоны")
        self.canvas_frame3 = ttk.Frame(self.tab_pheromone); self.canvas_frame3.pack(fill=tk.BOTH, expand=True)

    def _generate_cities(self, n: int) -> np.ndarray:
        return np.random.rand(n, 2) * 100.0

    def _on_generate_cities(self):
        self.cities = self._generate_cities(self.num_cities_var.get())
        self.distance_var.set("-"); self.iteration_var.set("-")
        self._draw_cities_preview()
        self._clear_convergence_tab()
        self._clear_pheromone_tab()
        self.status_var.set("Города сгенерированы, можно запускать поиск")

    def _draw_cities_preview(self):
        for w in self.canvas_frame1.winfo_children(): w.destroy()
        fig = Figure(figsize=(7, 7), dpi=100); ax = fig.add_subplot(111)
        ax.scatter(self.cities[:, 0], self.cities[:, 1], c="red", s=90, zorder=4, edgecolors="black")
        for i, (x, y) in enumerate(self.cities):
            ax.annotate(str(i), (x, y), fontsize=9, ha="center", va="center", color="white", fontweight="bold")
        ax.set_title("Сгенерированные города (предпросмотр)")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.grid(True, alpha=0.3); ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        FigureCanvasTkAgg(fig, master=self.canvas_frame1).get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _clear_convergence_tab(self):
        for w in self.convergence_frame.winfo_children(): w.destroy()
        self.convergence_tree = None
        # не обнуляем self.convergence_results, чтобы графики могли использовать кэш

    def _clear_pheromone_tab(self):
        for w in self.canvas_frame3.winfo_children(): w.destroy()

    def _parse_float_range(self, text: str) -> List[float]:
        text = text.strip()
        if not text: return []
        vals: List[float] = []
        parts = [p.strip() for p in text.split(",")] if "," in text else [text]
        for p in parts:
            if ":" in p:
                seg = [s.strip() for s in p.split(":")]
                if len(seg) == 2:
                    start, stop = float(seg[0]), float(seg[1]); step = 0.1
                elif len(seg) == 3:
                    start, stop, step = float(seg[0]), float(seg[1]), float(seg[2])
                else:
                    continue
                x = start
                while x <= stop + 1e-12:
                    vals.append(round(x, 10)); x += step
            else:
                vals.append(float(p))
        seen = set(); out = []
        for v in vals:
            if v not in seen:
                seen.add(v); out.append(v)
        return out

    def _parse_int_range(self, text: str) -> List[int]:
        floats = self._parse_float_range(text)
        seen = set(); out = []
        for f in [int(round(v)) for v in floats]:
            if f not in seen:
                seen.add(f); out.append(f)
        return out

    def _run_algorithm(self):
        if self.is_running or self.is_converging:
            messagebox.showwarning("Внимание", "Выполняется другая операция"); return
        if self.cities is None:
            messagebox.showinfo("Требуется генерация", "Сначала «Сгенерировать города», затем «Запустить»."); return
        self.is_running = True; self.status_var.set("Выполнение...")
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

    def _run_convergence_study(self):
        if self.is_running or self.is_converging:
            messagebox.showwarning("Внимание", "Выполняется другая операция"); return
        if self.cities is None:
            messagebox.showinfo("Требуется генерация",
                                "Сначала «Сгенерировать города», затем «Исследовать на сходимость»."); return
        try:
            ants_list = self._parse_int_range(self.study_ants_entry.get()) or [30]
            alphas    = self._parse_float_range(self.study_alpha_entry.get()) or [0.5, 1.0]
            betas     = self._parse_float_range(self.study_beta_entry.get())  or [1.0, 2.0, 3.0]
            rhos      = self._parse_float_range(self.study_rho_entry.get())   or [0.3, 0.6]
            qs_list   = self._parse_int_range(self.study_q_entry.get())       or [100, 200]
        except Exception as e:
            messagebox.showerror("Ошибка ввода", f"Не удалось разобрать диапазоны: {e}"); return

        combos = len(ants_list) * len(alphas) * len(betas) * len(rhos) * len(qs_list)
        if combos > 1000:
            if not messagebox.askyesno("Подтверждение", f"Будет выполнено {combos} комбинаций. Продолжить?"):
                return

        def worker():
            try:
                self.is_converging = True
                self.status_var.set("Исследование сходимости...")
                rows: List[List[float]] = []
                iters = self.iterations_var.get()
                for ants in ants_list:
                    for a in alphas:
                        for b in betas:
                            for rho in rhos:
                                for Q in qs_list:
                                    algo = AntAlgorithmTSP(
                                        cities=self.cities,
                                        num_ants=ants,
                                        alpha=a, beta=b, rho=rho, Q=Q,
                                        iterations=iters
                                    )
                                    _, best_len = algo.run()
                                    rows.append([ants, a, b, rho, Q, round(float(best_len), 12)])
                self.convergence_results = rows
                self.root.after(0, lambda: self._draw_convergence_table(rows))
                self.root.after(0, lambda: self.status_var.set("Готово: таблица сформирована"))
            finally:
                self.is_converging = False

        threading.Thread(target=worker, daemon=True).start()

    def _draw_convergence_table(self, rows: List[List[float]]):
        for w in self.convergence_frame.winfo_children(): w.destroy()
        cols = ["Количество муравьёв", "Alpha", "Beta", "Evaporation", "Q", "Лучшее расстояние"]
        tree = ttk.Treeview(self.convergence_frame, columns=cols, show="headings", height=22)
        for c in cols:
            tree.heading(c, text=c)
            width = 180 if c == "Лучшее расстояние" else 150
            tree.column(c, width=width, anchor="center")
        for r in rows:
            tree.insert("", tk.END, values=r)
        tree.pack(fill=tk.BOTH, expand=True)
        self.convergence_tree = tree

    # Чтение данных из видимой таблицы (fallback для графиков)
    def _read_rows_from_tree(self) -> List[List[float]]:
        rows: List[List[float]] = []
        if not self.convergence_tree:
            return rows
        for iid in self.convergence_tree.get_children():
            vals = self.convergence_tree.item(iid, "values")
            if not vals: continue
            try:
                ants = int(vals[0]); a = float(vals[1]); b = float(vals[2])
                rho = float(vals[3]); Q = int(vals[4]); best = float(vals[5])
                rows.append([ants, a, b, rho, Q, best])
            except Exception:
                continue
        return rows

    def _update_visualization(self, best_path: List[int], best_length: float):
        self.distance_var.set(f"{best_length:.2f}")
        self.iteration_var.set(str(self.algorithm.current_iteration))

        for w in self.canvas_frame1.winfo_children(): w.destroy()
        fig1 = Figure(figsize=(7, 7), dpi=100); ax1 = fig1.add_subplot(111)
        ax1.scatter(self.cities[:, 0], self.cities[:, 1], c="red", s=90, zorder=4, edgecolors="black")
        for i, (x, y) in enumerate(self.cities):
            ax1.annotate(str(i), (x, y), fontsize=9, ha="center", va="center", color="white", fontweight="bold")
        for (path, _L) in self.algorithm.current_ant_paths[:5]:
            cyc = path + [path[0]]; pts = self.cities[cyc]
            ax1.plot(pts[:, 0], pts[:, 1], color="gray", alpha=0.25, linewidth=0.8, zorder=1)
        cyc_best = best_path + [best_path[0]]; pts_best = self.cities[cyc_best]
        ax1.plot(pts_best[:, 0], pts_best[:, 1], "b-", linewidth=2.2, zorder=3,
                 label=f"Лучший маршрут: {best_length:.2f}")
        ax1.set_title("Города и найденный маршрут"); ax1.set_xlabel("X"); ax1.set_ylabel("Y")
        ax1.grid(True, alpha=0.3); ax1.set_aspect("equal", adjustable="box"); ax1.legend(loc="best")
        fig1.tight_layout()
        FigureCanvasTkAgg(fig1, master=self.canvas_frame1).get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Феромоны
        for w in self.canvas_frame3.winfo_children(): w.destroy()
        fig3 = Figure(figsize=(7, 7), dpi=100); ax3 = fig3.add_subplot(111)
        pheromone_log = np.log10(self.algorithm.pheromone + 1.0)
        im = ax3.imshow(pheromone_log, cmap="YlOrRd", aspect="auto")
        ax3.set_title("Распределение феромонов (log10)"); ax3.set_xlabel("Город j"); ax3.set_ylabel("Город i")
        cbar = fig3.colorbar(im, ax=ax3); cbar.set_label("log10(феромон)")
        fig3.tight_layout()
        FigureCanvasTkAgg(fig3, master=self.canvas_frame3).get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status_var.set("Завершено!")

    # -------- Графики по таблице --------
    def _show_convergence_plots(self):
        # приоритет — прочитать видимую таблицу, затем кэш
        rows = self._read_rows_from_tree()
        if not rows and self.convergence_results:
            rows = self.convergence_results
        if not rows:
            messagebox.showinfo("Нет данных", "Сначала выполните «Исследовать на сходимость»."); return

        ants_vals = sorted({r[0] for r in rows})
        alpha_vals = sorted({r[1] for r in rows})
        beta_vals  = sorted({r[2] for r in rows})
        rho_vals   = sorted({r[3] for r in rows})
        q_vals     = sorted({r[4] for r in rows})

        ants_fix = ants_vals[0]; rho_fix = rho_vals[0]; q_fix = q_vals[0]

        win = tk.Toplevel(self.root); win.title("Графики на основе таблицы"); win.geometry("1200x800")
        nb = ttk.Notebook(win); nb.pack(fill=tk.BOTH, expand=True)

        # vs Alpha
        tabA = ttk.Frame(nb); nb.add(tabA, text="Лучшее vs Alpha")
        figA = Figure(figsize=(6, 5), dpi=100); axA = figA.add_subplot(111)
        selA = [r for r in rows if r[0]==ants_fix and r[3]==rho_fix and r[4]==q_fix]
        selA.sort(key=lambda x: x[1])
        if selA:
            axA.plot([r[1] for r in selA], [r[5] for r in selA], "o-b")
        axA.set_xlabel("Alpha"); axA.set_ylabel("Лучшее расстояние")
        axA.grid(True, alpha=0.3); axA.set_title(f"ants={ants_fix}, rho={rho_fix}, Q={q_fix}")
        FigureCanvasTkAgg(figA, master=tabA).get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # vs Beta (линии по alpha)
        tabB = ttk.Frame(nb); nb.add(tabB, text="Лучшее vs Beta")
        figB = Figure(figsize=(6, 5), dpi=100); axB = figB.add_subplot(111)
        selB = [r for r in rows if r[0]==ants_fix and r[3]==rho_fix and r[4]==q_fix]
        for a in alpha_vals:
            line = [r for r in selB if r[1]==a]
            line.sort(key=lambda x: x[2])
            if line:
                axB.plot([r[2] for r in line], [r[5] for r in line], marker="o", label=f"alpha={a}")
        axB.set_xlabel("Beta"); axB.set_ylabel("Лучшее расстояние")
        axB.grid(True, alpha=0.3); axB.legend(); axB.set_title(f"ants={ants_fix}, rho={rho_fix}, Q={q_fix}")
        FigureCanvasTkAgg(figB, master=tabB).get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # vs Rho (линии по alpha)
        tabR = ttk.Frame(nb); nb.add(tabR, text="Лучшее vs ρ")
        figR = Figure(figsize=(6, 5), dpi=100); axR = figR.add_subplot(111)
        selR = [r for r in rows if r[0]==ants_fix and r[4]==q_fix]
        for a in alpha_vals:
            line = [r for r in selR if r[1]==a]
            line.sort(key=lambda x: x[3])
            if line:
                axR.plot([r[3] for r in line], [r[5] for r in line], marker="o", label=f"alpha={a}")
        axR.set_xlabel("Evaporation (ρ)"); axR.set_ylabel("Лучшее расстояние")
        axR.grid(True, alpha=0.3); axR.legend(); axR.set_title(f"ants={ants_fix}, Q={q_fix}")
        FigureCanvasTkAgg(figR, master=tabR).get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Heatmap Alpha×Beta
        tabH = ttk.Frame(nb); nb.add(tabH, text="Heatmap Alpha×Beta")
        figH = Figure(figsize=(6, 5), dpi=100); axH = figH.add_subplot(111)
        grid = np.full((len(alpha_vals), len(beta_vals)), np.nan)
        for i, a in enumerate(alpha_vals):
            for j, b in enumerate(beta_vals):
                match = [r[5] for r in rows if r[0]==ants_fix and r[1]==a and r[2]==b and r[3]==rho_fix and r[4]==q_fix]
                if match:
                    grid[i, j] = match[0]
        im = axH.imshow(grid, cmap="viridis", aspect="auto", origin="lower")
        axH.set_xticks(range(len(beta_vals))); axH.set_xticklabels([str(b) for b in beta_vals])
        axH.set_yticks(range(len(alpha_vals))); axH.set_yticklabels([str(a) for a in alpha_vals])
        axH.set_xlabel("Beta"); axH.set_ylabel("Alpha")
        axH.set_title(f"Лучшее расстояние (ants={ants_fix}, rho={rho_fix}, Q={q_fix})")
        figH.colorbar(im, ax=axH, label="Лучшее расстояние")
        figH.tight_layout()
        FigureCanvasTkAgg(figH, master=tabH).get_tk_widget().pack(fill=tk.BOTH, expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = AntVisualizationApp(root)
    root.mainloop()
