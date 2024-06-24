import numpy as np


class AdaptiveStochasticTunneling:
    def __init__(self, budget, dim=5, pop_size=50, F=0.8, CR=0.9, alpha=0.5, gamma=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover probability
        self.alpha = alpha  # Scaling factor for tunneling function
        self.gamma = gamma  # Curvature parameter for tunneling function
        self.bounds = (-5.0, 5.0)

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], size=(self.pop_size, self.dim))

    def tunnel_fitness(self, fitness, best_f):
        # Transform fitness using a tunneling function to escape local minima
        return best_f - self.alpha * np.exp(-self.gamma * (fitness - best_f))

    def mutate(self, population, idx):
        indices = [i for i in range(self.pop_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = np.clip(
            population[a] + self.F * (population[b] - population[c]), self.bounds[0], self.bounds[1]
        )
        return mutant

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, population, f_values, trial, trial_f, trial_idx, best_f):
        transformed_trial_f = self.tunnel_fitness(trial_f, best_f)
        transformed_target_f = self.tunnel_fitness(f_values[trial_idx], best_f)
        if transformed_trial_f < transformed_target_f:
            population[trial_idx] = trial
            f_values[trial_idx] = trial_f

    def __call__(self, func):
        population = self.initialize_population()
        f_values = np.array([func(ind) for ind in population])
        n_evals = self.pop_size
        best_f = np.min(f_values)

        while n_evals < self.budget:
            for idx in range(self.pop_size):
                mutant = self.mutate(population, idx)
                trial = self.crossover(population[idx], mutant)
                trial_f = func(trial)
                n_evals += 1
                self.select(population, f_values, trial, trial_f, idx, best_f)
                if n_evals >= self.budget:
                    break
            best_f = np.min(f_values)  # Update best found solution

        self.f_opt = np.min(f_values)
        self.x_opt = population[np.argmin(f_values)]

        return self.f_opt, self.x_opt
