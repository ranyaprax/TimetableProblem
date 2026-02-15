from sched import scheduler
from tsp_ga import *
import numpy as np
import optuna
import numpy as np

# # if __name__ == "__main__":
# #     random.seed(42)
# #     def generate_enrollment_matrix(rows, cols, enrollment_prob=0.5):
# #         E = []
# #         for _ in range(rows):
# #             row = [1 if random.random() < enrollment_prob else 0 for _ in range(cols)]
# #             E.append(row)
# #         return E

# #     def print_enrollment_matrix(E):
# #         print("E = [")
# #         for row in E:
# #             print("    " + str(row) + ",")
# #         print("]")
# #     def create_instance_file(filename):
# #         K = 30   # number of time slots
# #         M = 10  # rows (students)
# #         N = 30  # columns (exams)

# #         E = generate_enrollment_matrix(M, N, enrollment_prob=0.5)
# #         print_enrollment_matrix(E)

# #         with open(filename, "w") as f:
# #             f.write(f"{M} {N} {K}\n")
# #             for row in E:
# #                 f.write(" ".join(map(str, row)) + "\n")

# #     create_instance_file("instance.txt")
#     # scheduler = ExamSchedulerGA(
#     #     filename="medium-1.txt",
#     #     population_size=100,
#     #     generations=500,
#     #     crossover_rate=0.8,
#     #     mutation_rate=0.10,
#     #     tournament_size=10,
#     #     hard_penalty=10,
#     #     soft_weight=1,
#     #     use_student_based_ops=True # Configuration for class-based mutation/cross-over - better performance observed
#     # )
#     # scheduler.run_ga()

# def objective(trial):

#     # ---- Hyperparameter Search Space ----
#     population_size = trial.suggest_int("population_size", 50, 300, step=50)
#     generations = trial.suggest_int("generations", 200, 800, step=100)
#     crossover_rate = trial.suggest_float("crossover_rate", 0.6, 0.95)
#     mutation_rate = trial.suggest_float("mutation_rate", 0.01, 0.3)
#     tournament_size = trial.suggest_int("tournament_size", 3, 20)
#     use_student_based_ops = trial.suggest_categorical(
#         "use_student_based_ops", [True, False]
#     )

#     # ---- Repeat runs because GA is stochastic ----
#     runs = 3
#     scores = []

#     for _ in range(runs):
#         scheduler = ExamSchedulerGA(
#             filename="medium-1.txt",
#             population_size=population_size,
#             generations=generations,
#             crossover_rate=crossover_rate,
#             mutation_rate=mutation_rate,
#             tournament_size=tournament_size,
#             hard_penalty=10,
#             soft_weight=1,
#             use_student_based_ops=use_student_based_ops
#         )

#         result = scheduler.run_ga()

#         # Multi-objective collapsed to single score
#         score = result["hard_violations"] * 1000 + result["soft_cost"]
#         scores.append(score)

#     return np.mean(scores)


# study = optuna.create_study(direction="minimize")
# study.optimize(objective, n_trials=50)

# for key, value in study.best_trial.params.items():
#     print(f"{key}: {value}")

#     # ---- Re-run GA with best parameters and enable plotting ----
# best_scheduler = ExamSchedulerGA(
#     filename="medium-1.txt",
#     population_size=study.best_trial.params["population_size"],
#     generations=study.best_trial.params["generations"],
#     crossover_rate=study.best_trial.params["crossover_rate"],
#     mutation_rate=study.best_trial.params["mutation_rate"],
#     tournament_size=study.best_trial.params["tournament_size"],
#     hard_penalty=10,
#     soft_weight=1,
#     use_student_based_ops=study.best_trial.params["use_student_based_ops"]
# )

# print("\nBest trial:")
# print("Value:", study.best_trial.value)
# print("Params:")
# for key, value in study.best_trial.params.items():
#     print(f"  {key}: {value}")
    
# optuna.visualization.plot_optimization_history(study)
# optuna.visualization.plot_param_importances(study)


best_scheduler = ExamSchedulerGA(
    filename="medium-1.txt",
    population_size=200,
    generations=800,
    crossover_rate=0.925870125847676,
    mutation_rate=0.06898817448743613,
    tournament_size=8,
    hard_penalty=10,
    soft_weight=1,
    use_student_based_ops=False,
    plot=True
)


best_result = best_scheduler.run_ga()

print("\nFinal Best Run Results:")
print("Hard Violations:", best_result["hard_violations"])
print("Soft Cost:", best_result["soft_cost"])
