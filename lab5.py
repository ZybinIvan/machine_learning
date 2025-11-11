import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class MedianFinderGA:
    """Генетический алгоритм для поиска вершин-медиан графа"""

    def __init__(self, distance_matrix, num_medians=3, population_size=50,
                 crossover_prob=0.7, mutation_prob=0.1, max_generations=100):
        self.distance_matrix = np.array(distance_matrix)
        self.num_vertices = len(distance_matrix)
        self.num_medians = num_medians
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_generations = max_generations
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def initialize_population(self):
        """Инициализация начальной популяции"""
        population = []
        for _ in range(self.population_size):
            # Создаем хромосому - список из num_medians уникальных вершин
            chromosome = sorted(random.sample(range(self.num_vertices), self.num_medians))
            population.append(chromosome)
        return population

    def fitness_function(self, chromosome):
        """
        Функция приспособленности: минимизируем сумму расстояний
        от каждой вершины до ближайшей медианы
        """
        total_distance = 0
        medians = set(chromosome)

        for vertex in range(self.num_vertices):
            if vertex not in medians:
                # Находим минимальное расстояние до одной из медиан
                min_dist = min(self.distance_matrix[vertex][median]
                              for median in chromosome)
                total_distance += min_dist

        # Возвращаем инверсию для максимизации (чем меньше расстояние, тем лучше)
        return 1.0 / (1.0 + total_distance)

    def calculate_total_distance(self, chromosome):
        """Вычисляет общее расстояние для хромосомы"""
        total_distance = 0
        medians = set(chromosome)

        for vertex in range(self.num_vertices):
            if vertex not in medians:
                min_dist = min(self.distance_matrix[vertex][median]
                              for median in chromosome)
                total_distance += min_dist

        return total_distance

    def selection(self, population, fitness_values):
        """Селекция методом рулетки"""
        total_fitness = sum(fitness_values)
        probabilities = [f / total_fitness for f in fitness_values]

        # Выбираем родителей
        selected = random.choices(population, weights=probabilities,
                                 k=self.population_size)
        return selected

    def crossover(self, parent1, parent2):
        """Скрещивание с сохранением уникальности вершин"""
        if random.random() > self.crossover_prob:
            return parent1.copy(), parent2.copy()

        # Упорядоченное скрещивание
        crossover_point = random.randint(1, self.num_medians - 1)

        child1 = parent1[:crossover_point]
        child2 = parent2[:crossover_point]

        # Добавляем недостающие элементы
        for gene in parent2:
            if gene not in child1 and len(child1) < self.num_medians:
                child1.append(gene)

        for gene in parent1:
            if gene not in child2 and len(child2) < self.num_medians:
                child2.append(gene)

        # Если не хватает генов, добавляем случайные
        available1 = [i for i in range(self.num_vertices) if i not in child1]
        available2 = [i for i in range(self.num_vertices) if i not in child2]

        while len(child1) < self.num_medians:
            child1.append(random.choice(available1))
            available1.remove(child1[-1])

        while len(child2) < self.num_medians:
            child2.append(random.choice(available2))
            available2.remove(child2[-1])

        return sorted(child1), sorted(child2)

    def mutation(self, chromosome):
        """Мутация: замена случайной вершины"""
        if random.random() > self.mutation_prob:
            return chromosome

        mutated = chromosome.copy()
        mutation_index = random.randint(0, self.num_medians - 1)

        # Найти доступные вершины
        available = [i for i in range(self.num_vertices) if i not in mutated]
        if available:
            mutated[mutation_index] = random.choice(available)
            mutated = sorted(mutated)

        return mutated

    def run(self, callback=None):
        """Запуск генетического алгоритма"""
        # Инициализация
        population = self.initialize_population()
        best_solution = None
        best_fitness = 0

        for generation in range(self.max_generations):
            # Вычисление приспособленности
            fitness_values = [self.fitness_function(chromo) for chromo in population]

            # Сохранение лучшего решения
            max_fitness_idx = fitness_values.index(max(fitness_values))
            if fitness_values[max_fitness_idx] > best_fitness:
                best_fitness = fitness_values[max_fitness_idx]
                best_solution = population[max_fitness_idx].copy()

            # Сохранение статистики
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(sum(fitness_values) / len(fitness_values))

            # Обратный вызов для обновления интерфейса
            if callback:
                callback(generation, best_solution, best_fitness)

            # Селекция
            selected = self.selection(population, fitness_values)

            # Создание нового поколения
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = selected[i]
                parent2 = selected[min(i + 1, self.population_size - 1)]

                # Скрещивание
                child1, child2 = self.crossover(parent1, parent2)

                # Мутация
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

        return best_solution, self.calculate_total_distance(best_solution)


class MedianFinderGUI:
    """Графический интерфейс для генетического алгоритма поиска медиан"""

    def __init__(self, root):
        self.root = root
        self.root.title("Поиск вершин-медиан графа (Генетический алгоритм)")
        self.root.geometry("1200x800")

        self.distance_matrix = None
        self.ga = None
        self.is_running = False

        self.create_widgets()
        self.generate_random_matrix()

    def create_widgets(self):
        """Создание элементов интерфейса"""
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Левая панель (параметры и управление)
        left_frame = ttk.LabelFrame(main_frame, text="Параметры", padding="10")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Параметры
        ttk.Label(left_frame, text="Число вершин (N):").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.num_vertices_var = tk.IntVar(value=20)
        ttk.Entry(left_frame, textvariable=self.num_vertices_var, width=10).grid(row=0, column=1, pady=5)

        ttk.Label(left_frame, text="Число медиан:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.num_medians_var = tk.IntVar(value=3)
        ttk.Entry(left_frame, textvariable=self.num_medians_var, width=10).grid(row=1, column=1, pady=5)

        ttk.Label(left_frame, text="Размер популяции:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.pop_size_var = tk.IntVar(value=50)
        ttk.Entry(left_frame, textvariable=self.pop_size_var, width=10).grid(row=2, column=1, pady=5)

        ttk.Label(left_frame, text="Вероятность скрещивания:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.crossover_var = tk.DoubleVar(value=0.7)
        ttk.Entry(left_frame, textvariable=self.crossover_var, width=10).grid(row=3, column=1, pady=5)

        ttk.Label(left_frame, text="Вероятность мутации:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.mutation_var = tk.DoubleVar(value=0.1)
        ttk.Entry(left_frame, textvariable=self.mutation_var, width=10).grid(row=4, column=1, pady=5)

        ttk.Label(left_frame, text="Макс. поколений:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.max_gen_var = tk.IntVar(value=100)
        ttk.Entry(left_frame, textvariable=self.max_gen_var, width=10).grid(row=5, column=1, pady=5)

        # Кнопки управления
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="Сгенерировать матрицу",
                  command=self.generate_random_matrix).pack(pady=5, fill=tk.X)
        ttk.Button(button_frame, text="Показать матрицу",
                  command=self.show_matrix).pack(pady=5, fill=tk.X)
        self.run_button = ttk.Button(button_frame, text="Запустить ГА",
                                     command=self.run_algorithm)
        self.run_button.pack(pady=5, fill=tk.X)

        # Панель результатов
        result_frame = ttk.LabelFrame(left_frame, text="Результаты", padding="10")
        result_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        self.result_text = scrolledtext.ScrolledText(result_frame, width=40, height=15,
                                                     wrap=tk.WORD, font=("Courier", 9))
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Правая панель (графики)
        right_frame = ttk.LabelFrame(main_frame, text="Визуализация процесса эволюции",
                                     padding="10")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)

        # Matplotlib figure
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Настройка весов для изменения размера
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def generate_random_matrix(self):
        """Генерация случайной матрицы расстояний"""
        n = self.num_vertices_var.get()

        # Генерация симметричной матрицы расстояний
        self.distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distance = random.randint(1, 100)
                self.distance_matrix[i][j] = distance
                self.distance_matrix[j][i] = distance

        messagebox.showinfo("Успех", f"Матрица расстояний {n}x{n} сгенерирована!")
        self.log_message(f"Сгенерирована матрица расстояний {n}x{n}")

    def show_matrix(self):
        """Показать матрицу расстояний"""
        if self.distance_matrix is None:
            messagebox.showwarning("Предупреждение", "Сначала сгенерируйте матрицу!")
            return

        matrix_window = tk.Toplevel(self.root)
        matrix_window.title("Матрица расстояний")
        matrix_window.geometry("600x600")

        text_widget = scrolledtext.ScrolledText(matrix_window, wrap=tk.NONE,
                                               font=("Courier", 8))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Форматированный вывод матрицы
        n = len(self.distance_matrix)
        text_widget.insert(tk.END, "    " + "  ".join(f"{i:3d}" for i in range(n)) + "\n")
        text_widget.insert(tk.END, "   " + "-" * (n * 5) + "\n")

        for i in range(n):
            row_str = f"{i:2d} |"
            for j in range(n):
                row_str += f"{int(self.distance_matrix[i][j]):4d} "
            text_widget.insert(tk.END, row_str + "\n")

        text_widget.config(state=tk.DISABLED)

    def run_algorithm(self):
        """Запуск генетического алгоритма"""
        if self.distance_matrix is None:
            messagebox.showwarning("Предупреждение", "Сначала сгенерируйте матрицу!")
            return

        if self.is_running:
            messagebox.showwarning("Предупреждение", "Алгоритм уже выполняется!")
            return

        self.is_running = True
        self.run_button.config(state=tk.DISABLED)
        self.result_text.delete(1.0, tk.END)

        # Создание экземпляра ГА
        self.ga = MedianFinderGA(
            distance_matrix=self.distance_matrix,
            num_medians=self.num_medians_var.get(),
            population_size=self.pop_size_var.get(),
            crossover_prob=self.crossover_var.get(),
            mutation_prob=self.mutation_var.get(),
            max_generations=self.max_gen_var.get()
        )

        self.log_message("=== ЗАПУСК ГЕНЕТИЧЕСКОГО АЛГОРИТМА ===")
        self.log_message(f"Популяция: {self.pop_size_var.get()}")
        self.log_message(f"Поколений: {self.max_gen_var.get()}")
        self.log_message(f"Вероятность скрещивания: {self.crossover_var.get()}")
        self.log_message(f"Вероятность мутации: {self.mutation_var.get()}")
        self.log_message("")

        # Запуск алгоритма с обратным вызовом
        def callback(generation, best_solution, best_fitness):
            if generation % 10 == 0:
                self.log_message(f"Поколение {generation}: Медианы = {best_solution}, "
                               f"Приспособленность = {best_fitness:.6f}")
                self.update_plot()
                self.root.update()

        best_solution, total_distance = self.ga.run(callback=callback)

        # Вывод результатов
        self.log_message("")
        self.log_message("=== РЕЗУЛЬТАТЫ ===")
        self.log_message(f"Найденные медианы: {best_solution}")
        self.log_message(f"Суммарное расстояние: {total_distance:.2f}")
        self.log_message("")

        # Подробная информация о медианах
        self.log_message("Детальная информация о медианах:")
        for i, median in enumerate(best_solution, 1):
            self.log_message(f"  Медиана {i}: вершина {median}")

        # Вычисление расстояний от каждой вершины до ближайшей медианы
        self.log_message("")
        self.log_message("Расстояния от вершин до ближайших медиан:")
        medians_set = set(best_solution)
        for vertex in range(len(self.distance_matrix)):
            if vertex not in medians_set:
                min_dist = min(self.distance_matrix[vertex][median]
                             for median in best_solution)
                closest_median = min(best_solution,
                                    key=lambda m: self.distance_matrix[vertex][m])
                self.log_message(f"  Вершина {vertex} -> Медиана {closest_median} "
                               f"(расстояние: {min_dist:.2f})")

        self.update_plot()
        self.is_running = False
        self.run_button.config(state=tk.NORMAL)

    def update_plot(self):
        """Обновление графиков эволюции"""
        if self.ga is None or not self.ga.best_fitness_history:
            return

        self.figure.clear()

        # График приспособленности
        ax1 = self.figure.add_subplot(211)
        generations = range(len(self.ga.best_fitness_history))
        ax1.plot(generations, self.ga.best_fitness_history, 'b-',
                label='Лучшая приспособленность', linewidth=2)
        ax1.plot(generations, self.ga.avg_fitness_history, 'r--',
                label='Средняя приспособленность', linewidth=1)
        ax1.set_xlabel('Поколение')
        ax1.set_ylabel('Приспособленность')
        ax1.set_title('Эволюция популяции')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # График улучшения решения (инверсия приспособленности = расстояние)
        ax2 = self.figure.add_subplot(212)
        distances = [1.0 / f - 1.0 for f in self.ga.best_fitness_history]
        ax2.plot(generations, distances, 'g-', linewidth=2)
        ax2.set_xlabel('Поколение')
        ax2.set_ylabel('Суммарное расстояние')
        ax2.set_title('Минимизация суммарного расстояния')
        ax2.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def log_message(self, message):
        """Вывод сообщения в текстовое поле результатов"""
        self.result_text.insert(tk.END, message + "\n")
        self.result_text.see(tk.END)
        self.root.update()


def main():
    root = tk.Tk()
    app = MedianFinderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
