import numpy as np
import matplotlib.pyplot as plt
import time
import random
from typing import List, Tuple, Dict
import re

class TSPGeneticAlgorithm:
    def __init__(self, cities: np.ndarray, pop_size: int = 100, 
                 crossover_rate: float = 0.8, mutation_rate: float = 0.02):
        self.cities = cities
        self.n_cities = len(cities)
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.distance_matrix = self._calculate_distance_matrix()
        
    def _calculate_distance_matrix(self) -> np.ndarray:
        n = self.n_cities
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((self.cities[i][0] - self.cities[j][0])**2 + 
                              (self.cities[i][1] - self.cities[j][1])**2)
                dist_matrix[i][j] = dist_matrix[j][i] = dist
        return dist_matrix
    
    def _fitness(self, tour: List[int]) -> float:
        total_dist = sum(self.distance_matrix[tour[i]][tour[(i+1) % self.n_cities]] 
                        for i in range(self.n_cities))
        return 1 / total_dist if total_dist > 0 else 0
    
    def _initialize_population(self) -> List[List[int]]:
        population = []
        for _ in range(self.pop_size):
            tour = list(range(self.n_cities))
            random.shuffle(tour)
            population.append(tour)
        return population
    
    def _tournament_selection(self, population: List[List[int]], k: int = 3) -> List[int]:
        tournament = random.sample(population, k)
        return max(tournament, key=self._fitness)
    
    def _roulette_selection(self, population: List[List[int]]) -> List[int]:
        fitness_scores = [self._fitness(tour) for tour in population]
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.choice(population)
        
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, fitness in enumerate(fitness_scores):
            current += fitness
            if current > pick:
                return population[i]
        return population[-1]
    
    def _order_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child1 = [-1] * size
        child1[start:end] = parent1[start:end]
        pointer = end
        for city in parent2[end:] + parent2[:end]:
            if city not in child1:
                child1[pointer % size] = city
                pointer += 1
        
        child2 = [-1] * size
        child2[start:end] = parent2[start:end]
        pointer = end
        for city in parent1[end:] + parent1[:end]:
            if city not in child2:
                child2[pointer % size] = city
                pointer += 1
                
        return child1, child2
    
    def _pmx_crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child1, child2 = parent1[:], parent2[:]
        
        for i in range(start, end):
            val1, val2 = child1[i], child2[i]
            child1[child1.index(val2)], child1[i] = child1[i], child1[child1.index(val2)]
            child2[child2.index(val1)], child2[i] = child2[i], child2[child2.index(val1)]
            
        return child1, child2
    
    def _swap_mutation(self, tour: List[int]) -> List[int]:
        mutated = tour[:]
        i, j = random.sample(range(len(tour)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated
    
    def _inversion_mutation(self, tour: List[int]) -> List[int]:
        mutated = tour[:]
        i, j = sorted(random.sample(range(len(tour)), 2))
        mutated[i:j+1] = reversed(mutated[i:j+1])
        return mutated
    
    def solve(self, generations: int = 1000, selection_method: str = 'tournament',
              crossover_method: str = 'order', mutation_method: str = 'swap') -> Dict:
        
        start_time = time.time()
        population = self._initialize_population()
        best_fitness_history = []
        avg_fitness_history = []
        
        selection_func = self._tournament_selection if selection_method == 'tournament' else self._roulette_selection
        crossover_func = self._order_crossover if crossover_method == 'order' else self._pmx_crossover
        mutation_func = self._swap_mutation if mutation_method == 'swap' else self._inversion_mutation
        
        for generation in range(generations):
            fitness_scores = [self._fitness(tour) for tour in population]
            best_fitness = max(fitness_scores)
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            
            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)
            
            new_population = []
            
            # Elitism - keep best individual
            best_individual = population[fitness_scores.index(best_fitness)]
            new_population.append(best_individual[:])
            
            while len(new_population) < self.pop_size:
                parent1 = selection_func(population)
                parent2 = selection_func(population)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = crossover_func(parent1, parent2)
                else:
                    child1, child2 = parent1[:], parent2[:]
                
                if random.random() < self.mutation_rate:
                    child1 = mutation_func(child1)
                if random.random() < self.mutation_rate:
                    child2 = mutation_func(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.pop_size]
        
        end_time = time.time()
        
        final_fitness_scores = [self._fitness(tour) for tour in population]
        best_tour = population[final_fitness_scores.index(max(final_fitness_scores))]
        best_distance = 1 / max(final_fitness_scores)
        
        return {
            'best_tour': best_tour,
            'best_distance': best_distance,
            'best_fitness_history': best_fitness_history,
            'avg_fitness_history': avg_fitness_history,
            'computation_time': end_time - start_time
        }

def load_tsplib(filename: str) -> np.ndarray:
    """Load TSPLIB format file"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find NODE_COORD_SECTION
    coord_start = None
    for i, line in enumerate(lines):
        if 'NODE_COORD_SECTION' in line:
            coord_start = i + 1
            break
    
    if coord_start is None:
        raise ValueError("NODE_COORD_SECTION not found")
    
    cities = []
    for line in lines[coord_start:]:
        line = line.strip()
        if line == 'EOF' or not line:
            break
        parts = line.split()
        if len(parts) >= 3:
            cities.append([float(parts[1]), float(parts[2])])
    
    return np.array(cities)

def run_experiments():
    """Run experiments with different parameters"""
    # Test datasets (you'll need to download these)
    datasets = {
        'berlin52': 'berlin52.tsp',
        'kroA100': 'kroA100.tsp', 
        'pr1002': 'pr1002.tsp'
    }
    
    # Parameter variations
    pop_sizes = [50, 100, 200]
    crossover_rates = [0.6, 0.8, 0.9]
    mutation_rates = [0.01, 0.02, 0.05]
    
    results = {}
    
    for dataset_name, filename in datasets.items():
        try:
            cities = load_tsplib(filename)
            print(f"\nTesting {dataset_name} ({len(cities)} cities)")
            
            # Baseline run
            ga = TSPGeneticAlgorithm(cities)
            result = ga.solve(generations=500)
            
            print(f"Best distance: {result['best_distance']:.2f}")
            print(f"Computation time: {result['computation_time']:.2f}s")
            
            # Plot fitness evolution
            plt.figure(figsize=(10, 6))
            plt.plot(result['best_fitness_history'], label='Best Fitness')
            plt.plot(result['avg_fitness_history'], label='Average Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.title(f'Fitness Evolution - {dataset_name}')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{dataset_name}_fitness.png')
            plt.show()
            
            results[dataset_name] = result
            
        except FileNotFoundError:
            print(f"Dataset {filename} not found. Please download from TSPLIB.")
    
    return results

if __name__ == "__main__":
    print("TSP Genetic Algorithm Implementation")
    print("===================================")
    
    # Create a small test case
    test_cities = np.random.rand(20, 2) * 100
    
    ga = TSPGeneticAlgorithm(test_cities, pop_size=50)
    result = ga.solve(generations=200)
    
    print(f"Test run completed:")
    print(f"Best distance: {result['best_distance']:.2f}")
    print(f"Computation time: {result['computation_time']:.2f}s")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(result['best_fitness_history'], label='Best')
    plt.plot(result['avg_fitness_history'], label='Average')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    tour = result['best_tour']
    tour_coords = test_cities[tour + [tour[0]]]
    plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'b-o')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Best Tour Found')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('tsp_results.png')
    plt.show()
    
    run_experiments()
