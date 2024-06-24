import numpy as np
from scipy.optimize import minimize


class DynamicQuantumDifferentialEvolutionWithLocalSearchAndRestart:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.population_size = 80
        self.initial_num_elites = 5
        self.alpha = 0.6
        self.beta = 0.4
        self.local_search_prob = 0.4
        self.epsilon = 1e-6
        self.CR = 0.9
        self.F = 0.8

    def random_bounds(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

    def local_search(self, x, func):
        result = minimize(func, x, method="L-BFGS-B", bounds=[(self.bounds[0], self.bounds[1])] * self.dim)
        return result.x, result.fun

    def quantum_update(self, x, elites, beta):
        p_best = elites[np.random.randint(len(elites))]
        u = np.random.uniform(0, 1, self.dim)
        v = np.random.uniform(-1, 1, self.dim)
        Q = beta * (p_best - x) * np.log(1 / u)
        return np.clip(x + Q * v, self.bounds[0], self.bounds[1])

    def adaptive_restart(self, population, fitness, func):
        std_dev = np.std(fitness)
        if std_dev < self.epsilon:
            population = np.array([self.random_bounds() for _ in range(self.population_size)])
            fitness = np.array([func(ind) for ind in population])
        return population, fitness

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        population = np.array([self.random_bounds() for _ in range(self.population_size)])
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        personal_bests = np.copy(population)
        personal_best_fits = np.copy(fitness)
        global_best = population[np.argmin(fitness)]
        global_best_fit = np.min(fitness)
        num_elites = self.initial_num_elites

        while evaluations < self.budget:
            for i in range(self.population_size):
                a, b, c = population[
                    np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                ]
                mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < personal_best_fits[i]:
                        personal_bests[i] = trial
                        personal_best_fits[i] = f_trial
                        if f_trial < global_best_fit:
                            global_best_fit = f_trial
                            global_best = trial
                            if f_trial < self.f_opt:
                                self.f_opt = f_trial
                                self.x_opt = trial

                if np.random.rand() < self.local_search_prob and evaluations < self.budget:
                    refined_trial, f_refined_trial = self.local_search(trial, func)
                    evaluations += 1
                    if f_refined_trial < fitness[i]:
                        population[i] = refined_trial
                        fitness[i] = f_refined_trial
                        if f_refined_trial < personal_best_fits[i]:
                            personal_bests[i] = refined_trial
                            personal_best_fits[i] = f_refined_trial
                            if f_refined_trial < global_best_fit:
                                global_best_fit = f_refined_trial
                                global_best = refined_trial
                                if f_refined_trial < self.f_opt:
                                    self.f_opt = f_refined_trial
                                    self.x_opt = refined_trial

            elite_particles = personal_bests[np.argsort(personal_best_fits)[:num_elites]]
            for i in range(self.population_size):
                trial = self.quantum_update(population[i], elite_particles, self.beta)
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < personal_best_fits[i]:
                        personal_bests[i] = trial
                        personal_best_fits[i] = f_trial
                        if f_trial < global_best_fit:
                            global_best_fit = f_trial
                            global_best = trial
                            if f_trial < self.f_opt:
                                self.f_opt = f_trial
                                self.x_opt = trial

            population, fitness = self.adaptive_restart(population, fitness, func)

            if evaluations % (self.population_size * 10) == 0:
                diversity = np.std(fitness)
                if diversity < 1e-3:
                    for j in range(num_elites):
                        elite, elite_fit = self.local_search(personal_bests[j], func)
                        evaluations += 1
                        if elite_fit < personal_best_fits[j]:
                            personal_bests[j] = elite
                            personal_best_fits[j] = elite_fit
                            if elite_fit < global_best_fit:
                                global_best_fit = elite_fit
                                global_best = elite
                                if elite_fit < self.f_opt:
                                    self.f_opt = elite_fit
                                    self.x_opt = elite

            num_elites = max(2, min(self.initial_num_elites, int(self.population_size / 10)))

        return self.f_opt, self.x_opt
