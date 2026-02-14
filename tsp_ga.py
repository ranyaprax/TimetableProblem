import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class ExamSchedulerGA:
    def __init__(
        self,
        filename,
        population_size=100,
        generations=500,
        crossover_rate=0.8,
        mutation_rate=0.10,
        tournament_size=10,
        hard_penalty=1000,
        soft_weight=1,
        use_student_based_ops=False
    ):
        # Instance attributes
        self.filename = filename
        self.POPULATION_SIZE = population_size
        self.GENERATIONS = generations
        self.CROSSOVER_RATE = crossover_rate
        self.MUTATION_RATE = mutation_rate
        self.TOURNAMENT_SIZE = tournament_size
        self.HARD_PENALTY = hard_penalty
        self.SOFT_WEIGHT = soft_weight
        self.use_student_based_ops = use_student_based_ops 

        # GA state tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_hard_history = []
        self.best_soft_history = []

        # Problem instance
        self.M, self.N, self.K, self.E = self.read_instance(filename)

    def read_instance(self, filename):
        with open(filename, "r") as f:
            N, K, M = map(int, f.readline().split())
            E = [list(map(int, f.readline().split())) for _ in range(M)]

        # Optional safety check
        for row in E:
            if len(row) != N:
                raise ValueError("Each student row must have exactly N exam columns.")

        return M, N, K, E


    def initialize_population(self):
        return [[random.randint(1, self.K) for _ in range(self.N)]
                for _ in range(self.POPULATION_SIZE)]

    def evaluate_fitness(self, solution):
        hard_violations = 0
        soft_cost = 0

        for i in range(self.M):
            seen = {}
            for j in range(self.N):
                if self.E[i][j] == 1:
                    slot = solution[j]
                    if slot in seen:
                        hard_violations += 1
                    else:
                        seen[slot] = True

        for i in range(self.M):
            slots = [solution[j] for j in range(self.N) if self.E[i][j] == 1]
            slots.sort()
            for k in range(len(slots) - 1):
                if slots[k + 1] - slots[k] == 1:
                    soft_cost += 1

        fitness = self.HARD_PENALTY * hard_violations + self.SOFT_WEIGHT * soft_cost
        return fitness, hard_violations, soft_cost

    def select_parents(self, population):
        tournament = random.sample(population, self.TOURNAMENT_SIZE)
        tournament.sort(key=lambda s: self.evaluate_fitness(s)[0])
        return tournament[0], tournament[1]

    # --- Crossover ---
    def crossover(self, parent1, parent2):
        if self.use_student_based_ops:
            return self.student_based_crossover(parent1, parent2)
        else:
            return self.standard_crossover(parent1, parent2)

    def standard_crossover(self, parent1, parent2):
        if random.random() > self.CROSSOVER_RATE:
            return parent1[:], parent2[:]
        point = random.randint(1, self.N - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def student_based_crossover(self, parent1, parent2):
        if random.random() > self.CROSSOVER_RATE:
            return parent1[:], parent2[:]

        # Choose a random student
        student_idx = random.randint(0, self.M - 1)

        # Get exams that student is enrolled in
        exam_indices = [j for j in range(self.N) if self.E[student_idx][j] == 1]

        child1 = parent1[:]
        child2 = parent2[:]

        # Swap all those exam slots
        for j in exam_indices:
            child1[j], child2[j] = parent2[j], parent1[j]

        return child1, child2

    # --- Mutation ---
    def mutate(self, solution):
        if self.use_student_based_ops:
            self.class_swap_mutation(solution)
        else:
            self.standard_mutation(solution)

    def standard_mutation(self, solution):
        for i in range(len(solution)):
            if random.random() < self.MUTATION_RATE:
                solution[i] = random.randint(1, self.K)

    def class_swap_mutation(self, solution):
        for student_idx in range(self.M):  # iterate over students
            if random.random() < self.MUTATION_RATE:
                # Find all exams this student is in
                exam_indices = [exam_idx for exam_idx in range(self.N) if self.E[student_idx][exam_idx] == 1]
                if len(exam_indices) > 1:
                    a, b = random.sample(exam_indices, 2)
                    solution[a], solution[b] = solution[b], solution[a]


    def visualize_best_solution(self, solution):
        timetable = np.full((self.N, self.K), np.nan)  # N exams x K timeslots

        for exam_idx, slot in enumerate(solution):
            if 1 <= slot <= self.K:
                timetable[exam_idx][slot - 1] = exam_idx + 1
            else:
                print(f"Warning: exam {exam_idx} has invalid slot {slot}")

        annot_matrix = np.full(timetable.shape, '', dtype=object)
        for i in range(self.N):
            for j in range(self.K):
                if not np.isnan(timetable[i, j]):
                    annot_matrix[i, j] = str(int(timetable[i, j]))

        plt.figure(figsize=(14, 8))
        sns.heatmap(
            timetable,
            annot=annot_matrix,
            fmt='',
            cmap="tab20",
            cbar=True,
            linewidths=0.8,
            linecolor='gray',
            square=False
        )

        plt.xlabel("Time Slot")
        plt.ylabel("Exam")
        plt.title("Exam Schedule (Best Solution)")
        plt.xticks(ticks=np.arange(self.K) + 0.5, labels=np.arange(1, self.K + 1))
        plt.yticks(ticks=np.arange(self.N) + 0.5, labels=np.arange(1, self.N + 1), rotation=0)
        plt.show()

    def run_ga(self):
        population = self.initialize_population()

        # Track best by fitness (existing behavior)
        best_fitness = float("inf")

        # NEW: Track best by least hard violations
        best_hard_solution = None
        best_hard_violations = float("inf")
        best_hard_soft_cost = None
        best_hard_generation = None

        for gen in range(self.GENERATIONS):
            population.sort(key=lambda s: self.evaluate_fitness(s)[0])

            best_fit, best_hard, best_soft = self.evaluate_fitness(population[0])

            self.best_fitness_history.append(best_fit)
            self.best_hard_history.append(best_hard)
            self.best_soft_history.append(best_soft)

            total_fitness = sum(self.evaluate_fitness(sol)[0] for sol in population)
            self.avg_fitness_history.append(total_fitness / len(population))

            # ----- NEW: Track least hard violations -----
            for sol in population:
                fitness, hard, soft = self.evaluate_fitness(sol)

                if hard < best_hard_violations or (
                    hard == best_hard_violations and fitness < best_fitness
                ):
                    best_hard_solution = sol[:]
                    best_hard_violations = hard
                    best_hard_soft_cost = soft
                    best_hard_generation = gen

            # Elitism: keep top 2
            new_population = population[:2]

            while len(new_population) < self.POPULATION_SIZE:
                p1, p2 = self.select_parents(population)
                c1, c2 = self.crossover(p1, p2)
                self.mutate(c1)
                self.mutate(c2)
                new_population.extend([c1, c2])

            population = new_population[:self.POPULATION_SIZE]
        
        return {
            "hard_violations": best_hard_violations,
            "soft_cost": best_hard_soft_cost
        }


        # # ---- PRINT BEST SOLUTION FOUND (LEAST HARD VIOLATIONS) ----
        # print("\nBest Solution (Least Hard Constraint Violations):")
        # print("Generation:", best_hard_generation)
        # print("Hard Violations:", best_hard_violations)
        # print("Soft Cost:", best_hard_soft_cost)
        # print("Solution:", best_hard_solution)

        # self.plot_results()
        # self.visualize_best_solution(best_hard_solution)


    def plot_results(self):
        plt.figure()
        plt.plot(self.best_fitness_history, label="Best Fitness")
        plt.plot(self.avg_fitness_history, label="Average Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title("Best vs Average Fitness Over Generations")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(self.best_fitness_history)
        plt.title("Best Fitness")

        plt.subplot(3, 1, 2)
        plt.plot(self.best_hard_history)
        plt.title("Hard Constraint Violations")

        plt.subplot(3, 1, 3)
        plt.plot(self.best_soft_history)
        plt.title("Soft Constraint Cost")

        plt.tight_layout()
        plt.show()
