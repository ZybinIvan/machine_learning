import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Словарь оборудования
EQUIPMENT = {
    0: "Асинхронный двигатель",
    1: "Трансформатор",
    2: "Электрический кабель",
    3: "Вакуумный выключатель",
    4: "Промежуточное реле",
    5: "Магнитный пускатель",
    6: "Токовое реле",
    7: "Трансформатор тока",
    8: "Контактор",
    9: "Реле времени",
    10: "Реле напряжения",
}


class ART1:
    def __init__(self, d, N, vigilance, beta):
        self.d, self.N, self.vigilance, self.beta = d, N, vigilance, beta
        self.prototypes, self.clusters, self.history = [], {}, []

    def magnitude(self, v):
        return sum(v)

    def bitwise_and(self, v1, v2):
        return [a & b for a, b in zip(v1, v2)]

    def similarity_test(self, idx, e):
        if idx >= len(self.prototypes):
            return 0
        p = self.prototypes[idx]
        and_res = self.bitwise_and(p, e)
        return self.magnitude(and_res) / (self.beta + self.magnitude(p))

    def vigilance_test(self, idx, e):
        if idx >= len(self.prototypes) or self.magnitude(e) == 0:
            return False
        p = self.prototypes[idx]
        and_res = self.bitwise_and(p, e)
        return self.magnitude(and_res) / self.magnitude(e) >= self.vigilance

    def update_prototype(self, idx, e):
        self.prototypes[idx] = self.bitwise_and(self.prototypes[idx], e)

    def add_vector(self, e, vid):
        if not self.prototypes:
            self.prototypes.append(e[:])
            self.clusters[0] = [vid]
            self.history.append(f"Вектор {vid}: создан кластер 0")
            return 0

        sims = sorted(
            [(self.similarity_test(i, e), i) for i in range(len(self.prototypes))],
            reverse=True,
        )
        for _, idx in sims:
            if self.vigilance_test(idx, e):
                self.update_prototype(idx, e)
                if idx not in self.clusters:
                    self.clusters[idx] = []
                self.clusters[idx].append(vid)
                self.history.append(f"Вектор {vid}: добавлен в кластер {idx}")
                return idx

        if len(self.prototypes) < self.N:
            new_idx = len(self.prototypes)
            self.prototypes.append(e[:])
            self.clusters[new_idx] = [vid]
            self.history.append(f"Вектор {vid}: создан кластер {new_idx}")
            return new_idx

        self.history.append(f"Вектор {vid}: отклонен (макс кластеров)")
        return -1

    def train(self, vectors, max_iter=10):
        for it in range(max_iter):
            changed = False
            for vid, e in enumerate(vectors):
                old_c = next((c for c, m in self.clusters.items() if vid in m), None)
                new_c = self.add_vector(e, vid)
                if new_c != old_c and new_c >= 0 and old_c is not None:
                    self.clusters[old_c].remove(vid)
                    changed = True
            if not changed:
                break
        self.history.append(f"Завершено ({it + 1} итераций)")


class ART1App:
    def __init__(self, root):
        self.root = root
        self.root.title("ART1 - Кластеризация оборудования")
        self.root.geometry("1000x800")
        self.art1, self.vectors, self.initialized = None, [], False
        self.d, self.N, self.vigilance, self.beta = 11, 10, 0.9, 1
        self.checkboxes = []
        self.setup_ui()

    def show_bar_chart(self):
        if not self.art1 or not self.art1.clusters:
            messagebox.showerror("Ошибка", "Сначала запустите кластеризацию")
            return
        clusters = self.art1.clusters
        labels = list(clusters.keys())
        values = [len(clusters[cid]) for cid in labels]

        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        ax.bar([str(lbl) for lbl in labels], values, color="steelblue")
        ax.set_xlabel("Кластер")
        ax.set_ylabel("Число объектов")
        ax.set_title("Распределение объектов по кластерам")

        chart_win = tk.Toplevel(self.root)
        chart_win.title("График кластеризации")
        canvas = FigureCanvasTkAgg(fig, master=chart_win)
        chart_win.geometry("900x600")
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    def setup_ui(self):
        # === ПАРАМЕТРЫ ===
        pf = ttk.LabelFrame(self.root, text="Параметры", padding=5)
        pf.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(pf, text="d:").grid(row=0, column=0)
        self.d_e = ttk.Entry(pf, width=3)
        self.d_e.insert(0, "11")
        self.d_e.grid(row=0, column=1)

        ttk.Label(pf, text="N:").grid(row=0, column=2)
        self.n_e = ttk.Entry(pf, width=3)
        self.n_e.insert(0, "10")
        self.n_e.grid(row=0, column=3)

        ttk.Label(pf, text="ρ:").grid(row=0, column=4)
        self.v_e = ttk.Entry(pf, width=3)
        self.v_e.insert(0, "0.9")
        self.v_e.grid(row=0, column=5)

        ttk.Label(pf, text="β:").grid(row=0, column=6)
        self.b_e = ttk.Entry(pf, width=3)
        self.b_e.insert(0, "1")
        self.b_e.grid(row=0, column=7)

        self.init_btn = ttk.Button(pf, text="ИНИЦИАЛИЗИРОВАТЬ", command=self.initialize)
        self.init_btn.grid(row=0, column=8, padx=10)
        self.status_l = ttk.Label(pf, text="⚫ не инициализирован", foreground="red")
        self.status_l.grid(row=0, column=9, padx=10)

        if_frame = ttk.LabelFrame(
            self.root, text="Выбор оборудования (признаков)", padding=5
        )
        if_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for i, (idx, name) in enumerate(EQUIPMENT.items()):
            row, col = i // 4, i % 4
            var = tk.BooleanVar()
            cb = ttk.Checkbutton(
                if_frame, text=f"[{idx}] {name}", variable=var, state=tk.DISABLED
            )
            cb.grid(row=row, column=col, sticky=tk.W, padx=5, pady=3)
            self.checkboxes.append((idx, var, cb))

        # Кнопки
        bf = ttk.Frame(if_frame)
        bf.grid(row=3, column=0, columnspan=4, pady=10)
        ttk.Button(bf, text="Добавить вектор", command=self.add_vector).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(
            bf,
            text="Очистить выбор",
            command=lambda: [v.set(False) for _, v, _ in self.checkboxes],
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="Пример", command=self.load_example).pack(
            side=tk.LEFT, padx=5
        )

        lf = ttk.LabelFrame(self.root, text="Загруженные векторы", padding=5)
        lf.pack(fill=tk.X, padx=5, pady=5)
        sb = ttk.Scrollbar(lf)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.vectors_t = tk.Text(lf, height=5, width=80, yscrollcommand=sb.set)
        self.vectors_t.pack(fill=tk.BOTH, expand=True)
        sb.config(command=self.vectors_t.yview)

        ttk.Button(lf, text="Очистить список", command=self.clear_vectors).pack(
            anchor=tk.E, padx=5, pady=3
        )

        rf = ttk.LabelFrame(self.root, text="Результаты", padding=5)
        rf.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        sb2 = ttk.Scrollbar(rf)
        sb2.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_t = scrolledtext.ScrolledText(
            rf, height=12, width=80, yscrollcommand=sb2.set
        )
        self.results_t.pack(fill=tk.BOTH, expand=True)
        sb2.config(command=self.results_t.yview)

        self.run_btn = ttk.Button(
            self.root, text="ЗАПУСТИТЬ ART1", command=self.run_art1, state=tk.DISABLED
        )
        self.run_btn.pack(pady=5)
        ttk.Button(
            self.root, text="График кластеров", command=self.show_bar_chart
        ).pack(pady=5)

    def initialize(self):
        try:
            self.d = int(self.d_e.get())
            self.N = int(self.n_e.get())
            self.vigilance = float(self.v_e.get())
            self.beta = int(self.b_e.get())

            if (
                not (0 < self.vigilance <= 1)
                or self.d <= 0
                or self.N <= 0
                or self.beta <= 0
            ):
                raise ValueError("Параметры некорректны")

            for e in [self.d_e, self.n_e, self.v_e, self.b_e]:
                e.config(state=tk.DISABLED)
            self.init_btn.config(state=tk.DISABLED)

            for _, _, cb in self.checkboxes:
                cb.config(state=tk.NORMAL)
            self.run_btn.config(state=tk.NORMAL)

            self.initialized = True
            self.status_l.config(
                text=f"🟢 инициализирован (d={self.d})", foreground="green"
            )
        except:
            messagebox.showerror("Ошибка", "Параметры некорректны")

    def add_vector(self):
        if not self.initialized:
            messagebox.showerror("Ошибка", "Инициализируйте алгоритм")
            return

        vector = [int(v.get()) for _, v, _ in self.checkboxes]
        if len(vector) != self.d:
            messagebox.showerror("Ошибка", f"Вектор должен быть размером {self.d}")
            return

        self.vectors.append(vector)
        for _, v, _ in self.checkboxes:
            v.set(False)
        self.display_vectors()

    def display_vectors(self):
        self.vectors_t.delete(1.0, tk.END)
        for i, v in enumerate(self.vectors):
            equip = [EQUIPMENT[j] for j in range(len(v)) if v[j] == 1]
            self.vectors_t.insert(tk.END, f"{i}: {equip if equip else '(пусто)'}\n")

    def run_art1(self):
        if not self.vectors:
            messagebox.showerror("Ошибка", "Загрузите векторы")
            return

        self.art1 = ART1(self.d, self.N, self.vigilance, self.beta)
        self.art1.train(self.vectors)
        self.show_results()
        messagebox.showinfo("Успех", "Алгоритм завершен")

    def show_results(self):
        self.results_t.delete(1.0, tk.END)
        res = {
            "clusters": self.art1.clusters,
            "prototypes": self.art1.prototypes,
            "history": self.art1.history,
        }

        self.results_t.insert(
            tk.END,
            f"{'=' * 90}\nПАРАМЕТРЫ: d={self.d}, N={self.N}, ρ={self.vigilance}, β={self.beta}\n{'=' * 90}\n\n",
        )

        for cid in sorted(res["clusters"].keys()):
            equip = [
                f"[{i}] {EQUIPMENT[i]}"
                for i, b in enumerate(res["prototypes"][cid])
                if b == 1
            ]
            self.results_t.insert(
                tk.END,
                f"Кластер {cid}:\n"
                f"  Прототип: {res['prototypes'][cid]}\n"
                f"  Оборудование: {', '.join(equip) if equip else '(нет)'}\n"
                f"  Векторы: {res['clusters'][cid]}\n\n",
            )

        self.results_t.insert(tk.END, f"{'=' * 90}\nИСТОРИЯ\n{'=' * 90}\n\n")
        for msg in res["history"][-25:]:
            self.results_t.insert(tk.END, msg + "\n")

    def clear_vectors(self):
        self.vectors = []
        self.display_vectors()
        self.results_t.delete(1.0, tk.END)

    def load_example(self):
        if not self.initialized:
            for e, v in [
                (self.d_e, 11),
                (self.n_e, 10),
                (self.v_e, 0.9),
                (self.b_e, 1),
            ]:
                e.delete(0, tk.END)
                e.insert(0, str(v))
            self.initialize()

        examples = [
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
            [1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        ]
        self.vectors = examples
        self.display_vectors()

if __name__ == '__main__':
    root = tk.Tk()
    app = ART1App(root)
    root.mainloop()
