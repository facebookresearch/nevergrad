import numpy as np


class EnhancedAdaptiveQGSA_v41:
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
        self.f_opt = np.Inf
        self.x_opt = None

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

    def _adaptive_parameters(self, current_fitness):
        self.G0 *= 0.95
        self.alpha *= 0.95
        if current_fitness < self.f_opt:
            self.delta = min(0.2, self.delta * 1.03)
        else:
            self.delta = max(0.05, self.delta * 0.97)
        self.f_opt = current_fitness

    def _update_agents(self, agents, new_agents, fitness_values, new_fitness_values):
        for i in range(self.num_agents):
            if new_fitness_values[i] < fitness_values[i]:
                agents[i] = new_agents[i]
                fitness_values[i] = new_fitness_values[i]
                if new_fitness_values[i] < self.f_opt:
                    self.f_opt = new_fitness_values[i]
                    self.x_opt = new_agents[i]

    def __call__(self, func):
        agents = self._initialize_agents()

        for _ in range(self.budget):
            fitness_values = np.array([self._objective_function(func, agent) for agent in agents])
            best_agent_idx = np.argmin(fitness_values)
            best_agent = agents[best_agent_idx]
            masses = self._calculate_masses(fitness_values)

            new_agents = np.copy(agents)
            new_fitness_values = np.copy(fitness_values)

            for i in range(self.num_agents):
                if i != best_agent_idx:
                    force = sum(
                        [
                            self._calculate_gravitational_force(agents[i], masses[i], best_agent)
                            for i in range(self.num_agents)
                            if i != best_agent_idx
                        ]
                    )
                    new_agents[i] = self._update_agent_position(agents[i], force)
                    new_fitness_values[i] = self._objective_function(func, new_agents[i])

            self._update_agents(agents, new_agents, fitness_values, new_fitness_values)
            self._adaptive_parameters(np.min(fitness_values))

        return self.f_opt, self.x_opt
