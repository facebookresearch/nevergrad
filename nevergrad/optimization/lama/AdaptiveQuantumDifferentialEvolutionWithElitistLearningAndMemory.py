import numpy as np
from scipy.optimize import minimize


class AdaptiveQuantumDifferentialEvolutionWithElitistLearningAndMemory:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.population_size = 100
        self.initial_num_elites = 5
        self.alpha = 0.5
        self.beta = 0.3
        self.local_search_prob = 0.7
        self.epsilon = 1e-6
        self.CR = 0.9
        self.F = 0.8
        self.diversity_threshold = 1e-3
        self.adaptive_restart_interval = 100
        self.memory_rate = 0.5
        self.learning_rate = 0.5
        self.num_learning_agents = 10

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

    def adaptive_num_elites(self, diversity):
        if diversity < self.diversity_threshold:
            return max(2, int(self.population_size / 20))
        else:
            return self.initial_num_elites

    def hybrid_local_search(self, x, func):
        methods = ["L-BFGS-B", "TNC"]
        f_best = np.inf
        x_best = None
        for method in methods:
            result = minimize(func, x, method=method, bounds=[(self.bounds[0], self.bounds[1])] * self.dim)
            if result.fun < f_best:
                f_best = result.fun
                x_best = result.x
        return x_best, f_best

    def elitist_learning(self, population, elites, func):
        new_population = np.copy(population)
        for i in range(self.num_learning_agents):
            elite = elites[np.random.randint(len(elites))]
            learner = np.copy(elite)
            perturbation = np.random.uniform(-self.learning_rate, self.learning_rate, self.dim)
            learner = np.clip(learner + perturbation, self.bounds[0], self.bounds[1])
            f_learner = func(learner)

            if f_learner < func(elite):
                new_population[i] = learner
            else:
                new_population[i] = elite

        return new_population

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        population = np.array([self.random_bounds() for _ in range(self.population_size)])
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        memory = np.copy(population)
        memory_fitness = np.copy(fitness)

        personal_bests = np.copy(population)
        personal_best_fits = np.copy(fitness)
        global_best = population[np.argmin(fitness)]
        global_best_fit = np.min(fitness)

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
                    refined_trial, f_refined_trial = self.hybrid_local_search(trial, func)
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

            diversity = np.std(fitness)
            num_elites = self.adaptive_num_elites(diversity)
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

            if evaluations % self.adaptive_restart_interval == 0:
                population, fitness = self.adaptive_restart(population, fitness, func)

            # Memory update
            for i in range(self.population_size):
                if fitness[i] < memory_fitness[i]:
                    memory[i] = population[i]
                    memory_fitness[i] = fitness[i]
                else:
                    trial = self.memory_rate * memory[i] + (1 - self.memory_rate) * population[i]
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

            # Elitist learning phase
            learned_population = self.elitist_learning(personal_bests, elite_particles, func)
            learned_fitness = np.array([func(ind) for ind in learned_population])
            evaluations += self.num_learning_agents

            for i in range(self.num_learning_agents):
                if learned_fitness[i] < global_best_fit:
                    global_best_fit = learned_fitness[i]
                    global_best = learned_population[i]
                    if learned_fitness[i] < self.f_opt:
                        self.f_opt = learned_fitness[i]
                        self.x_opt = learned_population[i]

        return self.f_opt, self.x_opt
