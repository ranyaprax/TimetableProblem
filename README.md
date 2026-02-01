
# Genetic Algorithm for Exam Scheduling

This project implements a **Genetic Algorithm (GA)** to solve a university exam scheduling problem. The goal is to assign exams to time slots while minimizing hard constraint violations (students having overlapping exams) and soft constraint costs (students having consecutive exams).

---

## Features

* Reads exam instances from a file containing:

  * Number of students
  * Number of exams
  * Number of available time slots
  * Enrollment matrix indicating which students take which exams
* Implements a GA with:

  * Tournament selection
  * Single-point crossover
  * Mutation
  * Elitism
* Tracks and plots:

  * Best fitness per generation
  * Average fitness per generation
  * Hard constraint violations
  * Soft constraint cost
* Automatically generates a sample exam instance if none is provided.

---

## Dependencies

* Python 3.7+
* `matplotlib` for plotting:

```bash
pip install matplotlib
```

---

## Usage

1. **Generate an exam instance** (optional):

The script automatically generates a random instance file `instance.txt` with 10 students, 30 exams, and 30 time slots. To customize, modify `create_instance_file()` parameters:

```python
K = 30   # Number of time slots
M = 10   # Number of students
N = 30   # Number of exams
```

2. **Run the GA**:

```bash
python __main__.py
```

The GA will:

* Initialize a random population of solutions
* Evolve solutions over a specified number of generations
* Plot fitness evolution and constraint statistics

---

## Configuration

Parameters for the GA are set at the top of the script:

```python
POPULATION_SIZE = 100
GENERATIONS = 500
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.10
TOURNAMENT_SIZE = 10

HARD_PENALTY = 10
SOFT_WEIGHT = 1
```

* **POPULATION_SIZE**: Number of candidate solutions per generation
* **GENERATIONS**: Number of generations to evolve
* **CROSSOVER_RATE**: Probability of crossover between two parents
* **MUTATION_RATE**: Probability of mutation per exam assignment
* **TOURNAMENT_SIZE**: Number of individuals in tournament selection
* **HARD_PENALTY**: Fitness penalty for hard constraint violations
* **SOFT_WEIGHT**: Fitness weight for soft constraint cost

---

## Files

* `instance.txt` — Input file for exam instance
* `python __main__.py` — Main GA implementation and instance generator

**Instance file format:**

```
M N K
E[0][0] E[0][1] ... E[0][N-1]
...
E[M-1][0] E[M-1][1] ... E[M-1][N-1]
```

* `M`: Number of students
* `N`: Number of exams
* `K`: Number of time slots
* `E`: Enrollment matrix (1 = student enrolled in exam, 0 = not enrolled)

---

## Output

* **Plots**:

  1. Best vs Average fitness over generations
  2. Best fitness per generation
  3. Hard constraint violations
  4. Soft constraint cost

* **Best solution** (optional print, currently commented out in code):

```python
# Best solution (exam → time slot)
# Fitness
# Hard constraint violations
# Soft constraint cost
```

---

## Notes

* The GA uses **elitism**, keeping the top 2 solutions each generation.
* Hard constraint violations are heavily penalized to prioritize feasible schedules.
* The algorithm is stochastic; results vary between runs. Use `random.seed()` for reproducibility.

