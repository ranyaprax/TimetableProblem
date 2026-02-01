import random
from tsp_ga import *

if __name__ == "__main__":
    random.seed(42)
    def generate_enrollment_matrix(rows, cols, enrollment_prob=0.5):
        E = []
        for _ in range(rows):
            row = [1 if random.random() < enrollment_prob else 0 for _ in range(cols)]
            E.append(row)
        return E

    def print_enrollment_matrix(E):
        print("E = [")
        for row in E:
            print("    " + str(row) + ",")
        print("]")
    def create_instance_file(filename):
        K = 30   # number of time slots
        M = 10  # rows (students)
        N = 30  # columns (exams)

        E = generate_enrollment_matrix(M, N, enrollment_prob=0.5)
        print_enrollment_matrix(E)

        with open(filename, "w") as f:
            f.write(f"{M} {N} {K}\n")
            for row in E:
                f.write(" ".join(map(str, row)) + "\n")

    create_instance_file("instance.txt")
    scheduler = ExamSchedulerGA(
        filename="instance.txt",
        population_size=100,
        generations=500,
        crossover_rate=0.8,
        mutation_rate=0.10,
        tournament_size=10,
        hard_penalty=10,
        soft_weight=1
    )
    scheduler.run_ga()
