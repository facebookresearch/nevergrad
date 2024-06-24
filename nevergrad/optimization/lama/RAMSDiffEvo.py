import numpy as np


class RAMSDiffEvo:
    def __init__(self, budget, population_size=100, F_base=0.8, CR_base=0.5, perturbation=0.05):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F_base = F_base  # Further increased base mutation factor for enhanced exploration
        self.CR_base = CR_base  # Lowered crossover probability to avoid premature convergence
        self.perturbation = perturbation  # Reduced perturbation for more stable adaptive parameters

    def __call__(self, func):
        # Initialize population and fitness evaluations
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(x) for x in population])
        num_evals = self.population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        while num_evals < self.budget:
            new_population = np.empty_like(population)

            for i in range(self.population_size):
                if num_evals >= self.budget:
                    break

                # Mutation with dynamic strategy adaptation based on previous improvements
                strategy_type = np.random.choice(
                    ["best1bin", "rand1bin", "rand2best1bin", "current2rand1"], p=[0.4, 0.2, 0.3, 0.1]
                )
                F = np.clip(self.F_base + self.perturbation * np.random.randn(), 0.1, 1.0)
                CR = np.clip(self.CR_base + self.perturbation * np.random.randn(), 0.0, 1.0)

                idxs = [idx for idx in range(self.population_size) if idx != i]
                chosen = np.random.choice(idxs, 3, replace=False)
                a, b, c = population[chosen]

                if strategy_type == "best1bin":
                    mutant = best_individual + F * (b - c)
                elif strategy_type == "rand1bin":
                    mutant = a + F * (b - c)
                elif strategy_type == "rand2best1bin":
                    mutant = a + F * (best_individual - a) + F * (b - c)
                else:  # 'current2rand1'
                    mutant = population[i] + F * (a - population[i] + b - c)

                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dimension) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                num_evals += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()
                else:
                    new_population[i] = population[i]

            population = new_population.copy()

        return best_fitness, best_individual
