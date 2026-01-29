import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict


class TPGeneticAlgorithm:
    def __init__(self, population_size: int, mutation_rate: float, crossover_rate: float, generations: int, selection_method: str):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.selection_method = selection_method

    def initialize_population(pop_size): # Ranya
        return None

    def evaluate_fitness(solution): # Ranya
        return None

    def __selection(self, population:  List[List[int]) -> List[int]: # Sade
        return None

    def __class_crossover(self, parent1: List[int], parent2: : List[int]) -> Tuple[List[int], List[int]]: # Sade
        return None

    def __class_mutation(self, solution: List[int]) -> List[int]: # Sade
        return None

    def run_ga():
        return None
    

