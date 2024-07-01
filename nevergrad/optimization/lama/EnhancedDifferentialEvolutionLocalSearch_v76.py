import numpy as np


class EnhancedDifferentialEvolutionLocalSearch_v76:
    def __init__(
        self,
        budget=10000,
        p_best=0.2,
        f_min=0.5,
        f_max=0.9,
        cr_min=0.5,
        cr_max=0.9,
        local_search_iters=1000,
        perturbation_factor=0.1,
        population_size=50,
    ):
        self.budget = budget
        self.dim = 5
        self.p_best = p_best
        self.f_min = f_min
        self.f_max = f_max
        self.cr_min = cr_min
        self.cr_max = cr_max
        self.local_search_iters = local_search_iters
        self.perturbation_factor = perturbation_factor
        self.population_size = population_size

    def enhanced_de_local_search(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        def mutate(population, target_idx, f):
            candidates = [idx for idx in range(len(population)) if idx != target_idx]
            a, b, c = population[np.random.choice(candidates, 3, replace=False)]
            return np.clip(a + f * (b - c), -5.0, 5.0)

        def crossover(trial, target, cr):
            mask = np.random.rand(self.dim) < cr
            if not np.any(mask):
                mask[np.random.randint(0, self.dim)] = True
            trial = np.where(np.random.rand(self.dim) < cr, trial, target)
            return trial

        for _ in range(self.budget):
            new_population = []
            for idx, target in enumerate(population):
                f = np.clip(np.random.normal(np.mean([self.f_min, self.f_max]), 0.1), self.f_min, self.f_max)
                cr = np.clip(
                    np.random.normal(np.mean([self.cr_min, self.cr_max]), 0.1), self.cr_min, self.cr_max
                )

                p_best_idxs = np.random.choice(
                    [i for i in range(self.population_size) if i != idx],
                    int(self.p_best * self.population_size),
                    replace=False,
                )
                if idx in p_best_idxs:
                    p_best_idx = np.random.choice([i for i in range(self.population_size) if i != idx])
                    p_best_target = population[p_best_idx]
                    trial = mutate(
                        [
                            p_best_target,
                            target,
                            population[
                                np.random.choice(
                                    [i for i in range(self.population_size) if i not in [idx, p_best_idx]]
                                )
                            ],
                        ],
                        1,
                        f,
                    )
                else:
                    trial = mutate(population, idx, f)

                new_trial = crossover(trial.copy(), target, cr)

                target_val = func(target)
                trial_val = func(trial)
                new_trial_val = func(new_trial)

                if new_trial_val <= target_val:
                    population[idx] = new_trial
                    if new_trial_val <= trial_val:
                        population[idx] = trial

            for idx, target in enumerate(population):
                for _ in range(self.local_search_iters):
                    perturbed = target + self.perturbation_factor * np.random.normal(0, 1, self.dim)
                    perturbed = np.clip(perturbed, -5.0, 5.0)
                    if func(perturbed) <= func(target):
                        population[idx] = perturbed

        best_idx = np.argmin([func(sol) for sol in population])
        best_solution = population[best_idx]
        best_fitness = func(best_solution)

        return best_fitness

    def __call__(self, func):
        best_fitness = self.enhanced_de_local_search(func)

        return best_fitness, np.random.uniform(
            -5.0, 5.0, self.dim
        )  # Return the best fitness and a random solution
