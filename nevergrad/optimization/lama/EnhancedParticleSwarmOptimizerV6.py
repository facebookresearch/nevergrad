import numpy as np


class EnhancedParticleSwarmOptimizerV6:
    def __init__(
        self,
        budget,
        swarm_size=20,
        inertia_weight=0.7,
        cognitive_weight=1.5,
        social_weight=1.5,
        max_velocity=0.2,
    ):
        self.budget = budget
        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.max_velocity = max_velocity

    def initialize_swarm(self, func):
        return [
            np.random.uniform(func.bounds.lb, func.bounds.ub, size=len(func.bounds.lb))
            for _ in range(self.swarm_size)
        ]

    def clipToBounds(self, vector, func):
        return np.clip(vector, func.bounds.lb, func.bounds.ub)

    def optimize_particle(self, particle, func, personal_best, global_best, velocity):
        new_velocity = (
            self.inertia_weight * velocity
            + self.cognitive_weight * np.random.rand() * (personal_best - particle)
            + self.social_weight * np.random.rand() * (global_best - particle)
        )
        new_velocity = np.clip(new_velocity, -self.max_velocity, self.max_velocity)

        new_particle = particle + new_velocity
        return self.clipToBounds(new_particle, func), new_velocity

    def __call__(self, func):
        swarm = self.initialize_swarm(func)
        personal_best = np.copy(swarm)
        global_best = np.copy(swarm[np.argmin([func(p) for p in swarm])])
        best_cost = func(global_best)

        for _ in range(self.budget):
            for i, particle in enumerate(swarm):
                velocity = np.zeros_like(particle)  # Initialize velocity for each particle
                swarm[i], velocity = self.optimize_particle(
                    particle, func, personal_best[i], global_best, velocity
                )
                personal_best[i] = np.where(
                    func(swarm[i]) < func(personal_best[i]), swarm[i], personal_best[i]
                )
                if func(swarm[i]) < best_cost:
                    global_best = np.copy(swarm[i])
                    best_cost = func(global_best)

        return best_cost, global_best
