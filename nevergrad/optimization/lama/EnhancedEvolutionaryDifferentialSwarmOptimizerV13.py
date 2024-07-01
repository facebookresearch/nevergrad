import numpy as np


class EnhancedEvolutionaryDifferentialSwarmOptimizerV13:
    def __init__(
        self,
        budget,
        swarm_size=50,
        differential_weight=0.8,
        crossover_rate=0.9,
        inertia_weight=0.6,
        cognitive_weight=1.5,
        social_weight=1.5,
        max_velocity=0.8,
        mutation_rate=0.05,
        num_generations=400,
        num_local_searches=1000,
    ):
        self.budget = budget
        self.swarm_size = swarm_size
        self.differential_weight = differential_weight
        self.crossover_rate = crossover_rate
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.max_velocity = max_velocity
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.num_local_searches = num_local_searches

    def initialize_swarm(self, func):
        return [
            np.random.uniform(func.bounds.lb, func.bounds.ub, size=len(func.bounds.lb))
            for _ in range(self.swarm_size)
        ]

    def clipToBounds(self, vector, func):
        return np.clip(vector, func.bounds.lb, func.bounds.ub)

    def optimize_particle(self, particle, func, personal_best, global_best, velocity, swarm):
        r1, r2 = np.random.choice(len(swarm), 2, replace=False)
        r3 = np.random.choice(len(swarm))

        mutant = swarm[r1] + self.differential_weight * (swarm[r2] - swarm[r3])
        crossover_mask = np.random.rand(len(particle)) < self.crossover_rate
        trial = np.where(crossover_mask, mutant, particle)

        new_velocity = (
            self.inertia_weight * velocity
            + self.cognitive_weight * np.random.rand() * (personal_best - particle)
            + self.social_weight * np.random.rand() * (global_best - particle)
        )
        new_velocity = np.clip(new_velocity, -self.max_velocity, self.max_velocity)

        new_particle = particle + new_velocity
        return self.clipToBounds(new_particle, func), new_velocity, trial

    def mutate_particle(self, particle, func):
        mutated_particle = particle + np.random.normal(0, self.mutation_rate, size=len(particle))
        return self.clipToBounds(mutated_particle, func)

    def local_search(self, solution, func):
        best_solution = np.copy(solution)
        best_cost = func(solution)

        for _ in range(self.num_local_searches):
            new_solution = self.mutate_particle(solution, func)
            new_cost = func(new_solution)

            if new_cost < best_cost:
                best_solution = np.copy(new_solution)
                best_cost = new_cost

        return best_solution

    def hybrid_optimization(self, func):
        swarm = self.initialize_swarm(func)
        personal_best = np.copy(swarm)
        global_best = np.copy(swarm[np.argmin([func(p) for p in swarm])])
        best_cost = func(global_best)

        for _ in range(self.num_generations):
            for i, particle in enumerate(swarm):
                velocity = np.zeros_like(particle)  # Initialize velocity for each particle
                swarm[i], velocity, trial = self.optimize_particle(
                    particle, func, personal_best[i], global_best, velocity, swarm
                )
                personal_best[i] = np.where(func(trial) < func(personal_best[i]), trial, personal_best[i])

                if np.random.rand() < self.mutation_rate:
                    swarm[i] = self.mutate_particle(swarm[i], func)

                local_best = self.local_search(swarm[i], func)
                if func(local_best) < best_cost:
                    global_best = np.copy(local_best)
                    best_cost = func(global_best)

        return best_cost, global_best

    def __call__(self, func):
        best_aocc = 0
        best_solution = None

        for _ in range(self.budget):
            cost, solution = self.hybrid_optimization(func)
            if cost > best_aocc:
                best_aocc = cost
                best_solution = solution

        return best_aocc, best_solution
