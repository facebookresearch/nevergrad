"""
Auther: LLM-SHADE Team

This file is to be used for GECCO 2025 competition on LLM designed EA.
"""

import os
import random

import math
import types

import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from .lshade import *

initial_v1 = """
import numpy as np
from typing import Tuple, List
def initialize_1(self, args=None):
    x = self.rng_initialization.uniform(self.initial_lower_boundary, self.initial_upper_boundary,
                                        size=(self.n_individuals, self.ndim_problem))  # population
    y = np.empty((self.n_individuals,))  # fitness
    for i in range(self.n_individuals):
        if self._check_terminations():
            break
        y[i] = self._evaluate_fitness(x[i], args)
    a = np.empty((0, self.ndim_problem))  # set of archived inferior solutions
    return x, y, a"""

crossover_v1 = """
import numpy as np
from typing import Tuple, List
def crossover_1(self, x_mu=None, x=None, r=None):
    x_cr = np.copy(x)
    p_cr = np.empty((self.n_individuals,))  # crossover probabilities
    for k in range(self.n_individuals):
        p_cr[k] = self.rng_optimization.normal(self.m_mu[r[k]], 0.1)
        p_cr[k] = np.minimum(np.maximum(p_cr[k], 0.0), 1.0)
        i_rand = self.rng_optimization.integers(self.ndim_problem)
        for i in range(self.ndim_problem):
            if (i == i_rand) or (self.rng_optimization.random() < p_cr[k]):
                x_cr[k, i] = x_mu[k, i]
    return x_cr, p_cr"""

######################### LLM Designed Part ##########################

initial_v2 = '''
import numpy as np
from typing import Tuple, List
def initialize_2(self, args=None):
    """
    Crossover the population individuals.

    Args:
        self: The instance of the class containing the mutation parameters and methods.
            - n_individuals: int, Number of individuals in the population.
            - ndim_problem: int, Dimension of the problem.
            - initial_lower_boundary: int, Lower boundary for the initial population.
            - initial_upper_boundary: int, Upper boundary for the initial population.
            - _check_terminations(): method, 
                    Method to check termination conditions, 
                    Return True if reach the terminations else False.
            - h: int, Length of historical memory.
            - max_function_evaluations: int, Maximum number of function evaluations.
            - initial_pop_size: int, Initial population size.
            - _n_generations: int, Current number of generations.
            - m_median: np.ndarray, Median values of Cauchy distribution, shape=(self.h,).
            - _evaluate_fitness(individual: np.ndarray, args): method, 
                    Method to evaluate the fitness of an individual, input individual shape=(self.ndim_problem,).
                    Return fitness value of the individual.
            - rng_optimization: Random number generator for optimization, self.rng_optimization = np.random.default_rng(self.seed_optimization).

    Returns:
        x: np.ndarray, The current population of individuals, shape=(self.n_individuals, self.ndim_problem).
        y: np.ndarray, The current fitness values of the population, shape=(self.n_individuals,).
        a: np.ndarray, The archive of inferior solutions, shape=(0, self.ndim_problem).
    """
    x = np.zeros((self.n_individuals, self.ndim_problem))
    
    # Generate prime-spiral seed points for Voronoi tessellation
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    n_seeds = min(max(3, int(np.cbrt(self.n_individuals))), len(primes))
    
    seeds = np.zeros((n_seeds, self.ndim_problem))
    domain_range = self.initial_upper_boundary - self.initial_lower_boundary
    
    # Create seed points using prime-based spiral positioning
    for s in range(n_seeds):
        for d in range(self.ndim_problem):
            prime_factor = primes[s % len(primes)]
            spiral_param = (prime_factor * s + d * primes[(s + d) % len(primes)]) % 1000 / 1000.0
            
            # Apply logarithmic spiral transformation
            log_factor = np.log(1 + spiral_param * np.e)
            angular_component = 2 * np.pi * spiral_param * prime_factor / 10.0
            
            # Transform to domain coordinates
            seeds[s, d] = self.initial_lower_boundary + (
                log_factor * np.cos(angular_component) * 0.5 + 0.5
            ) * domain_range
    
    # Assign individuals to Voronoi cells
    individuals_per_cell = self.n_individuals // n_seeds
    remainder_individuals = self.n_individuals % n_seeds
    cell_assignments = []
    
    for s in range(n_seeds):
        cell_size = individuals_per_cell + (1 if s < remainder_individuals else 0)
        cell_assignments.extend([s] * cell_size)
    
    self.rng_optimization.shuffle(cell_assignments)
    
    current_idx = 0
    
    # Generate individuals within each Voronoi cell
    for cell_id in range(n_seeds):
        seed_point = seeds[cell_id]
        cell_indices = [i for i, c in enumerate(cell_assignments) if c == cell_id]
        
        for local_idx, global_idx in enumerate(cell_indices):
            if current_idx >= self.n_individuals:
                break
                
            # Calculate logarithmic density weight
            density_progress = (local_idx + 1) / len(cell_indices)
            log_density = np.log(1 + density_progress * (np.e - 1)) / np.log(np.e)
            
            # Generate direction using tessellation-aware sampling
            tessellation_angles = np.zeros(self.ndim_problem)
            for d in range(self.ndim_problem):
                base_angle = 2 * np.pi * d / self.ndim_problem
                cell_perturbation = cell_id * np.pi / n_seeds
                tessellation_angles[d] = base_angle + cell_perturbation + local_idx * 0.1
            
            # Create position vector using tessellation geometry
            position_offset = np.zeros(self.ndim_problem)
            max_cell_radius = 0.4 * np.linalg.norm(domain_range) / np.sqrt(n_seeds)
            
            for d in range(self.ndim_problem):
                radial_distance = max_cell_radius * log_density
                angle_d = tessellation_angles[d]
                
                # Apply dimensional scaling
                dim_scale = 1.0 / np.sqrt(1 + d * 0.1)
                position_offset[d] = radial_distance * np.sin(angle_d) * dim_scale
            
            # Add tessellation noise
            noise_strength = 0.1 * (1 - log_density * 0.6)
            tessellation_noise = self.rng_optimization.laplace(0, noise_strength, self.ndim_problem)
            scaled_noise = tessellation_noise * domain_range
            
            # Compute candidate position
            x[current_idx] = seed_point + position_offset + scaled_noise
            
            # Geometric boundary reflection
            for d in range(self.ndim_problem):
                reflection_count = 0
                while (x[current_idx, d] < self.initial_lower_boundary or 
                       x[current_idx, d] > self.initial_upper_boundary) and reflection_count < 3:
                    
                    if x[current_idx, d] < self.initial_lower_boundary:
                        overshoot = self.initial_lower_boundary - x[current_idx, d]
                        x[current_idx, d] = self.initial_lower_boundary + overshoot * 0.8
                    elif x[current_idx, d] > self.initial_upper_boundary:
                        overshoot = x[current_idx, d] - self.initial_upper_boundary
                        x[current_idx, d] = self.initial_upper_boundary - overshoot * 0.8
                    
                    reflection_count += 1
                
                # Final boundary enforcement
                x[current_idx, d] = np.clip(x[current_idx, d], 
                                          self.initial_lower_boundary, 
                                          self.initial_upper_boundary)
            
            current_idx += 1
    
    # Centroid attraction refinement phase
    attraction_fraction = 0.2
    n_attracted = int(attraction_fraction * self.n_individuals)
    attracted_indices = self.rng_optimization.choice(self.n_individuals, n_attracted, replace=False)
    
    # Calculate geometric centroid of each cell
    cell_centroids = np.zeros((n_seeds, self.ndim_problem))
    for cell_id in range(n_seeds):
        cell_members = [i for i, c in enumerate(cell_assignments[:current_idx]) if c == cell_id]
        if cell_members:
            cell_centroids[cell_id] = np.mean(x[cell_members], axis=0)
        else:
            cell_centroids[cell_id] = seeds[cell_id]
    
    for idx in attracted_indices:
        assigned_cell = cell_assignments[idx]
        centroid = cell_centroids[assigned_cell]
        
        # Apply adaptive centroid attraction
        attraction_weight = self.rng_optimization.beta(2, 5)
        direction_to_centroid = centroid - x[idx]
        
        # Scale attraction by distance
        distance_to_centroid = np.linalg.norm(direction_to_centroid)
        if distance_to_centroid > 0:
            normalized_direction = direction_to_centroid / distance_to_centroid
            attraction_magnitude = attraction_weight * distance_to_centroid * 0.3
            x[idx] += normalized_direction * attraction_magnitude
        
        # Boundary enforcement
        x[idx] = np.clip(x[idx], self.initial_lower_boundary, self.initial_upper_boundary)
    
    # Evaluate fitness for all individuals
    y = np.zeros(self.n_individuals)
    for i in range(self.n_individuals):
        y[i] = self._evaluate_fitness(x[i], args)
        if self._check_terminations():
            break
    
    # Initialize empty archive
    a = np.zeros((0, self.ndim_problem))
    
    return x, y, a
'''

initial_v3 = '''
import numpy as np
from typing import Tuple, List
def initialize_3(self, args=None):
    """
    Crossover the population individuals.

    Args:
        self: The instance of the class containing the mutation parameters and methods.
            - n_individuals: int, Number of individuals in the population.
            - ndim_problem: int, Dimension of the problem.
            - initial_lower_boundary: int, Lower boundary for the initial population.
            - initial_upper_boundary: int, Upper boundary for the initial population.
            - _check_terminations(): method, 
                    Method to check termination conditions, 
                    Return True if reach the terminations else False.
            - h: int, Length of historical memory.
            - max_function_evaluations: int, Maximum number of function evaluations.
            - initial_pop_size: int, Initial population size.
            - _n_generations: int, Current number of generations.
            - m_median: np.ndarray, Median values of Cauchy distribution, shape=(self.h,).
            - _evaluate_fitness(individual: np.ndarray, args): method, 
                    Method to evaluate the fitness of an individual, input individual shape=(self.ndim_problem,).
                    Return fitness value of the individual.
            - rng_optimization: Random number generator for optimization, self.rng_optimization = np.random.default_rng(self.seed_optimization).

    Returns:
        x: np.ndarray, The current population of individuals, shape=(self.n_individuals, self.ndim_problem).
        y: np.ndarray, The current fitness values of the population, shape=(self.n_individuals,).
        a: np.ndarray, The archive of inferior solutions, shape=(0, self.ndim_problem).
    """
    import numpy as np
    
    # Generate Sobol sequence for quasi-random sampling
    def sobol_sequence(n_samples, n_dims):
        # Simple Sobol-like sequence implementation
        samples = np.zeros((n_samples, n_dims))
        for i in range(n_dims):
            base = 2 + i
            sequence = []
            for j in range(n_samples):
                val = 0.0
                frac = 1.0 / base
                n = j + 1
                while n > 0:
                    val += (n % base) * frac
                    n //= base
                    frac /= base
                sequence.append(val % 1.0)
            samples[:, i] = sequence
        return samples
    
    # Generate initial quasi-random samples
    sobol_samples = sobol_sequence(self.n_individuals // 2, self.ndim_problem)
    
    # Create population array
    x = np.zeros((self.n_individuals, self.ndim_problem))
    
    # Fill first half with Sobol samples
    boundary_range = self.initial_upper_boundary - self.initial_lower_boundary
    dynamic_scale = 0.8 + 0.4 * np.exp(-self.ndim_problem / 20.0)
    
    for i in range(self.n_individuals // 2):
        x[i] = self.initial_lower_boundary + sobol_samples[i] * boundary_range * dynamic_scale
    
    # Generate second half using clustering around promising regions
    remaining = self.n_individuals - self.n_individuals // 2
    
    # Create cluster centers based on problem characteristics
    n_clusters = min(5, max(2, self.ndim_problem // 3))
    cluster_centers = np.zeros((n_clusters, self.ndim_problem))
    
    for c in range(n_clusters):
        # Distribute clusters across search space
        cluster_position = (c + 1) / (n_clusters + 1)
        for d in range(self.ndim_problem):
            offset = self.rng_optimization.uniform(-0.2, 0.2)
            cluster_centers[c, d] = self.initial_lower_boundary + (cluster_position + offset) * boundary_range
    
    # Assign remaining individuals to clusters
    cluster_size = remaining // n_clusters
    idx = self.n_individuals // 2
    
    for c in range(n_clusters):
        members = cluster_size if c < n_clusters - 1 else remaining - c * cluster_size
        
        for m in range(members):
            if idx < self.n_individuals:
                # Generate around cluster center with adaptive radius
                radius = 0.1 * boundary_range * (1.0 + 0.5 * self.rng_optimization.uniform())
                perturbation = self.rng_optimization.normal(0, radius, self.ndim_problem)
                x[idx] = cluster_centers[c] + perturbation
                idx += 1
    
    # Apply historical memory influence if available
    if hasattr(self, 'm_median') and len(self.m_median) > 0:
        memory_influence = 0.05 * np.mean(np.abs(self.m_median))
        memory_directions = self.rng_optimization.normal(0, memory_influence, x.shape)
        x += memory_directions
    
    # Ensure all individuals are within boundaries
    x = np.clip(x, self.initial_lower_boundary, self.initial_upper_boundary)
    
    # Evaluate fitness for all individuals
    y = np.zeros(self.n_individuals)
    for i in range(self.n_individuals):
        y[i] = self._evaluate_fitness(x[i], args)
    
    # Initialize empty archive
    a = np.zeros((0, self.ndim_problem))
    
    return x, y, a'''

crossover_v2 = '''
import numpy as np
from typing import Tuple, List
def crossover_2(self, x_mu=None, x=None, r=None):
    """
    Crossover the population individuals.

    Args:
        self: The instance of the class containing the mutation parameters and methods.
            - n_individuals: int, Number of individuals in the population.
            - ndim_problem: int, Dimension of the problem.
            - h: int, Length of historical memory.
            - p_min: int, Minimum population size, self.p_min = 2/self.n_individuals.
            - max_function_evaluations: int, Maximum number of function evaluations.
            - initial_pop_size: int, Initial population size.
            - _n_generations: int, Current number of generations.
            - m_median: np.ndarray, Median values of Cauchy distribution, shape=(self.h,).
            - rng_optimization: Random number generator for optimization, self.rng_optimization = np.random.default_rng(self.seed_optimization).
        x_mu: The mutated population of individuals, shape=(self.n_individuals, self.ndim_problem).
        x: The current population of individuals, shape=(self.n_individuals, self.ndim_problem).
        r: The indices of the selected individuals used for mutation and crossover, shape=(self.n_individuals,).

    Returns:
        x_cr: The crossover population of individuals, shape=(self.n_individuals, self.ndim_problem).
        p_cr: The crossover probabilities for each individual, shape=(self.n_individuals,).
    """
    import numpy as np
    from scipy.spatial.distance import cdist

    x_cr = np.copy(x_mu)
    p_cr = np.zeros(self.n_individuals)

    bracket_size = max(4, self.n_individuals // 4)
    n_brackets = max(2, self.n_individuals // bracket_size)

    fitness_ranks = np.argsort(np.argsort(r))
    centroid = np.mean(x_mu, axis=0)
    distances = np.linalg.norm(x_mu - centroid, axis=1)
    distance_ranks = np.argsort(np.argsort(distances))

    total_generations = self.max_function_evaluations / self.initial_pop_size
    cycle_period = total_generations * 0.2
    phase = (self._n_generations % cycle_period) / cycle_period

    oscillation_factor = 0.5 * (1 + np.sin(2 * np.pi * phase))
    exploration_weight = 0.8 * oscillation_factor + 0.2 * (1 - oscillation_factor)

    combined_scores = (1 - exploration_weight) * fitness_ranks + exploration_weight * distance_ranks
    sorted_indices = np.argsort(combined_scores)

    brackets = [[] for _ in range(n_brackets)]
    for i, idx in enumerate(sorted_indices):
        bracket_idx = i % n_brackets if (i // n_brackets) % 2 == 0 else n_brackets - 1 - (i % n_brackets)
        brackets[bracket_idx].append(idx)

    momentum = 0.5 * (1 + np.tanh((self._n_generations - total_generations * 0.3) / (total_generations * 0.1)))

    for bracket_idx, bracket in enumerate(brackets):
        if len(bracket) < 2:
            continue

        bracket_fitness = r[bracket]
        competitiveness = np.std(bracket_fitness) / (np.mean(bracket_fitness) + 1e-8)

        bracket_ranks = np.argsort(bracket_fitness)
        winners = [bracket[i] for i in bracket_ranks[:len(bracket)//2]]
        losers = [bracket[i] for i in bracket_ranks[len(bracket)//2:]]

        for i in range(len(bracket)):
            individual_idx = bracket[i]

            if individual_idx in winners:
                partner_pool = [w for w in winners if w != individual_idx]
                base_prob = 0.15 + 0.4 * (1 - momentum)
            else:
                partner_pool = winners
                base_prob = 0.65 + 0.15 * momentum

            if partner_pool:
                partner = self.rng_optimization.choice(partner_pool)
            else:
                partner = self.rng_optimization.choice([idx for idx in bracket if idx != individual_idx])

            oscillation_bonus = 0.1 * np.sin(4 * np.pi * phase)
            competitiveness_factor = np.tanh(2 * competitiveness)
            p_cr[individual_idx] = base_prob * (0.85 + 0.3 * competitiveness_factor) + oscillation_bonus
            p_cr[individual_idx] *= (0.75 + 0.25 * self.rng_optimization.random())
            p_cr[individual_idx] = np.clip(p_cr[individual_idx], 0.1, 0.9)

            mask = self.rng_optimization.random(self.ndim_problem) < p_cr[individual_idx]
            if not np.any(mask):
                mask[self.rng_optimization.integers(0, self.ndim_problem)] = True

            x_cr[individual_idx][mask] = x[partner][mask]

    return x_cr, p_cr'''


def lshade_optimize(evaluate, dim, budget):
    problem = {
        "fitness_function": evaluate,
        "ndim_problem": dim,
        "upper_boundary": 1.0,
        "lower_boundary": 0.0,
        "initial_upper_boundary": 1.0,
        "initial_lower_boundary": -1.0,
        #        'problem_name': f'Problem_{problem_index}',
        #        'rotation': Rotation,
        #        'lambda_': Lambda,
        #        'omega_': Omega
    }
    options = {
        "max_function_evaluations": budget,
        #        'fitness_threshold': OptimumValue
    }

    de = LSHADE(problem, options, 42, "useless")
    all_globals_namespace = {}

    exec(initial_v1, all_globals_namespace)
    exec(initial_v2, all_globals_namespace)
    exec(initial_v3, all_globals_namespace)
    exec(crossover_v2, all_globals_namespace)
    exec(crossover_v1, all_globals_namespace)
    program_callable_ini_v1 = all_globals_namespace["initialize_1"]
    program_callable_ini_v2 = all_globals_namespace["initialize_2"]
    program_callable_ini_v3 = all_globals_namespace["initialize_3"]
    program_callable_crossover_v2 = all_globals_namespace["crossover_2"]
    program_callable_crossover_v1 = all_globals_namespace["crossover_1"]

    de.initialize_1 = types.MethodType(program_callable_ini_v1, de)
    de.initialize_2 = types.MethodType(program_callable_ini_v2, de)
    de.initialize_3 = types.MethodType(program_callable_ini_v3, de)

    de.crossover_1 = types.MethodType(program_callable_crossover_v1, de)
    de.crossover_2 = types.MethodType(program_callable_crossover_v2, de)

    de.optimize()
