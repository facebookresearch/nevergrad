import numpy as np


class EnhancedAdaptiveQGSA_v25:
    def __init__(
        self, budget=1000, num_agents=30, G0=100.0, alpha=0.1, delta=0.1, lb=-5.0, ub=5.0, dimension=5
    ):
        self.budget = budget
        self.num_agents = num_agents
        self.G0 = G0
        self.alpha = alpha
        self.delta = delta
        self.lb = lb
        self.ub = ub
        self.dimension = dimension
        self.iteration = 0
        self.f_opt = np.Inf
        self.x_opt = None
        self.prev_best_fitness = np.Inf

    def _initialize_agents(self):
        return np.random.uniform(self.lb, self.ub, size=(self.num_agents, self.dimension))

    def _calculate_masses(self, fitness_values):
        return 1 / (fitness_values - np.min(fitness_values) + 1e-10)

    def _calculate_gravitational_force(self, agent, mass, best_agent):
        return self.G0 * mass * (best_agent - agent)

    def _update_agent_position(self, agent, force):
        new_pos = agent + self.alpha * force
        return np.clip(new_pos, self.lb, self.ub)

    def _objective_function(self, func, x):
        return func(x)

    def _adaptive_parameters(self):
        self.G0 *= 0.95
        self.alpha *= 0.96
        if self.f_opt < self.prev_best_fitness:
            self.delta = min(0.2, self.delta * 1.03)
        else:
            self.delta = max(0.05, self.delta * 0.97)
        self.prev_best_fitness = self.f_opt

    def _update_best_agent(self, agents, fitness_values):
        best_agent_idx = np.argmin(fitness_values)
        best_agent = agents[best_agent_idx]
        return best_agent, best_agent_idx

    def __call__(self, func):
        agents = self._initialize_agents()

        for _ in range(self.budget):
            fitness_values = np.array([self._objective_function(func, agent) for agent in agents])
            best_agent, best_agent_idx = self._update_best_agent(agents, fitness_values)
            masses = self._calculate_masses(fitness_values)

            for i in range(self.num_agents):
                if i != best_agent_idx:
                    force = sum(
                        [
                            self._calculate_gravitational_force(agents[i], masses[i], best_agent)
                            for i in range(self.num_agents)
                            if i != best_agent_idx
                        ]
                    )
                    agents[i] = self._update_agent_position(agents[i], force)
                    agents[i] = np.clip(agents[i], self.lb, self.ub)
                    fitness_values[i] = self._objective_function(func, agents[i])

                    if fitness_values[i] < self.f_opt:
                        self.f_opt = fitness_values[i]
                        self.x_opt = agents[i]

            self._adaptive_parameters()

        return self.f_opt, self.x_opt
