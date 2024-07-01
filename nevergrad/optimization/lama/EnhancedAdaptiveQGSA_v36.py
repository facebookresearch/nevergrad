import numpy as np


class EnhancedAdaptiveQGSA_v36:
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
        self.prev_best_fitness = np.Inf
        self.step_size = (ub - lb) * 0.1
        self.best_agents = []

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
        self.alpha *= 0.95
        if self.f_opt < self.prev_best_fitness:
            self.delta = min(0.2, self.delta * 1.03)
        else:
            self.delta = max(0.05, self.delta * 0.97)
        self.prev_best_fitness = self.f_opt

    def _update_best_agent(self, agents, fitness_values):
        best_agent_idx = np.argmin(fitness_values)
        best_agent = agents[best_agent_idx]
        return best_agent, best_agent_idx

    def _adjust_agent_position(self, agent, best_agent):
        r = np.random.uniform(-self.step_size, self.step_size, size=self.dimension)
        return np.clip(agent + r * (best_agent - agent), self.lb, self.ub)

    def _update_best_agents(self, agents, fitness_values, num_best_agents=2):
        best_agents_idx = np.argsort(fitness_values)[:num_best_agents]
        best_agents = agents[best_agents_idx]
        return best_agents

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
                    new_agent = self._update_agent_position(agents[i], force)
                    new_agent = self._adjust_agent_position(new_agent, best_agent)
                    new_fitness = self._objective_function(func, new_agent)

                    if new_fitness < fitness_values[i]:
                        agents[i] = new_agent
                        fitness_values[i] = new_fitness

                        if new_fitness < self.f_opt:
                            self.f_opt = new_fitness
                            self.x_opt = new_agent

            self._adaptive_parameters()
            self.best_agents.append(self._update_best_agents(agents, fitness_values))

        best_agents = self._update_best_agents(agents, fitness_values, num_best_agents=1)
        self.x_opt = best_agents[0]
        self.f_opt = self._objective_function(func, self.x_opt)

        return self.f_opt, self.x_opt
