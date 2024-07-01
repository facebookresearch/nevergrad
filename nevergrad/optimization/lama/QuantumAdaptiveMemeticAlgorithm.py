import numpy as np


class QuantumAdaptiveMemeticAlgorithm:
    def __init__(self, budget, population_size=50, tau1=0.1, tau2=0.1, memetic_rate=0.3, learning_rate=0.01):
        self.budget = budget
        self.population_size = population_size
        self.tau1 = tau1
        self.tau2 = tau2
        self.memetic_rate = memetic_rate
        self.learning_rate = learning_rate

    def gradient_estimation(self, func, x, h=1e-6):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x1 = np.copy(x)
            x2 = np.copy(x)
            x1[i] += h
            x2[i] -= h
            grad[i] = (func(x1) - func(x2)) / (2 * h)
        return grad

    def quantum_walk(self, x, global_best, alpha=0.1):
        return np.clip(x + alpha * (global_best - x) * np.random.uniform(-1, 1, size=x.shape), -5.0, 5.0)

    def evolutionary_step(self, func, pop, scores, crossover_rates, mutation_factors):
        new_pop = np.copy(pop)
        new_scores = np.copy(scores)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            F = mutation_factors[i]
            mutant = np.clip(a + F * (b - c), -5.0, 5.0)
            cross_points = np.random.rand(len(mutant)) < crossover_rates[i]
            if not np.any(cross_points):
                cross_points[np.random.randint(0, len(mutant))] = True
            trial = np.where(cross_points, mutant, pop[i])
            f = func(trial)
            if f < scores[i]:
                new_scores[i] = f
                new_pop[i] = trial
        return new_pop, new_scores

    def local_search(self, func, x, score):
        grad = self.gradient_estimation(func, x)
        candidate = np.clip(x - self.learning_rate * grad, -5.0, 5.0)
        f = func(candidate)
        if f < score:
            return candidate, f
        return x, score

    def adaptive_parameters(self, crossover_rates, mutation_factors):
        for i in range(self.population_size):
            if np.random.rand() < self.tau1:
                crossover_rates[i] = np.clip(crossover_rates[i] + np.random.normal(0, 0.1), 0, 1)
            if np.random.rand() < self.tau2:
                mutation_factors[i] = np.clip(mutation_factors[i] + np.random.normal(0, 0.1), 0, 2)

    def ensemble_step(self, func, pop, scores, crossover_rates, mutation_factors, global_best):
        new_pop, new_scores = self.evolutionary_step(func, pop, scores, crossover_rates, mutation_factors)
        for i in range(self.population_size):
            if np.random.rand() < self.memetic_rate:
                new_pop[i], new_scores[i] = self.local_search(func, new_pop[i], new_scores[i])
            else:
                new_pop[i] = self.quantum_walk(new_pop[i], global_best)
                new_scores[i] = func(new_pop[i])
        return new_pop, new_scores

    def temperature_schedule(self, current_iter, max_iter):
        return max(0.5, (1 - current_iter / max_iter))

    def __call__(self, func):
        np.random.seed(0)
        dim = 5
        lower_bound = -5.0
        upper_bound = 5.0

        # Initialize population
        pop = np.random.uniform(lower_bound, upper_bound, (self.population_size, dim))
        scores = np.array([func(ind) for ind in pop])

        # Initialize crossover rates and mutation factors
        crossover_rates = np.random.uniform(0.5, 1.0, self.population_size)
        mutation_factors = np.random.uniform(0.5, 1.0, self.population_size)

        # Global best initialization
        best_idx = np.argmin(scores)
        global_best_position = pop[best_idx]
        global_best_score = scores[best_idx]

        evaluations = self.population_size
        max_iterations = self.budget // self.population_size

        iteration = 0
        while evaluations < self.budget:
            # Adapt parameters
            self.adaptive_parameters(crossover_rates, mutation_factors)

            # Perform hybrid step
            pop, scores = self.ensemble_step(
                func, pop, scores, crossover_rates, mutation_factors, global_best_position
            )
            evaluations += self.population_size

            current_temp = self.temperature_schedule(iteration, max_iterations)
            self.learning_rate *= current_temp

            # Update global best from population
            best_idx = np.argmin(scores)
            if scores[best_idx] < global_best_score:
                global_best_score = scores[best_idx]
                global_best_position = pop[best_idx]

            iteration += 1

        self.f_opt = global_best_score
        self.x_opt = global_best_position
        return self.f_opt, self.x_opt
