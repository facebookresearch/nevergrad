from typing import Callable, List
from abc import ABC
import nevergrad
import numpy as np
from nevergrad.parametrization import parameter as p


class ConstrainedMultiObjective(ABC):
    def __init__(
        self,
        name: str,
        parametrization: p.Data,
        objective_funcs: List[Callable[[np.ndarray], float]],
        constraint_funcs: List[Callable[[np.ndarray], bool]],
        constraints_as_layer: bool = False,
    ) -> None:
        super().__init__()
        self.name: str = name
        self.parametrization: p.Data = parametrization
        self.objective_funcs: List[Callable[[np.ndarray], float]] = objective_funcs
        self.constraint_funcs: List[Callable[[np.ndarray], bool]] = constraint_funcs
        self.constraints_as_layer: bool = constraints_as_layer
        self._register_cheap_constraints()

    def _register_cheap_constraints(self) -> None:
        for f in self.constraint_funcs:
            self.parametrization.register_cheap_constraint(f, self.constraints_as_layer)

    def __call__(self, x: np.ndarray) -> List[float]:
        objectives: List[float] = [f(x) for f in self.objective_funcs]
        return objectives


class BinhKorn(ConstrainedMultiObjective):
    def __init__(self, constraints_as_layer: bool = False) -> None:
        super().__init__(
            name = "Binh and Korn function",
            parametrization = p.Array(shape=(2,), lower=np.array([0, 0]), upper=np.array([5, 3])),
            objective_funcs = [self.objective1, self.objective2],
            constraint_funcs = [self.constraint1, self.constraint2],
            constraints_as_layer = constraints_as_layer,
        )

    def objective1(self, x: np.ndarray) -> float:
        return float(4 *  np.sum(np.power(x, 2)))

    def objective2(self, x: np.ndarray) -> float:
        return float(np.sum(np.power(x + np.array([-5, -5]), 2)))

    def constraint1(self, x: np.ndarray) -> bool:
        return np.sum(np.power(x + np.array([-5, 0]), 2)) <= 25

    def constraint2(self, x: np.ndarray) -> bool:
        return np.sum(np.power(x + np.array([-8, 3]), 2)) >= 7.7


class ChankongHaimes(ConstrainedMultiObjective):
    def __init__(self, constraints_as_layer: bool = False) -> None:
        super().__init__(
            name = "Chankong and Haimes function",
            parametrization = p.Array(shape=(2,), lower=np.array([-20, -20]), upper=np.array([20, 20])),
            objective_funcs = [self.objective1, self.objective2],
            constraint_funcs = [self.constraint1, self.constraint2],
            constraints_as_layer = constraints_as_layer,
        )

    def objective1(self, x: np.ndarray) -> float:
        return float(np.sum(np.power(x + np.array([-2, -1]), 2)) + 2)

    def objective2(self, x: np.ndarray) -> float:
        return float(9 * x[0] - (x[1] - 1) ** 2)

    def constraint1(self, x: np.ndarray) -> bool:
        return np.sum(np.power(x, 2)) <= 225

    def constraint2(self, x: np.ndarray) -> bool:
        return x[0] - 3 * x[1] + 10 <= 0


class Test4(ConstrainedMultiObjective):
    def __init__(self, constraints_as_layer: bool = False) -> None:
        super().__init__(
            name = "Test function 4",
            parametrization = p.Array(shape=(2,), lower=np.array([-7, -7]), upper=np.array([4, 4])),
            objective_funcs = [self.objective1, self.objective2],
            constraint_funcs = [self.constraint1, self.constraint2, self.constraint3],
            constraints_as_layer = constraints_as_layer,
        )

    def objective1(self, x: np.ndarray) -> float:
        return float(x[0] ** 2 - x[1])

    def objective2(self, x: np.ndarray) -> float:
        return float(-0.5 * x[0] - x[1] - 1)

    def constraint1(self, x: np.ndarray) -> bool:
        return 6.5 - x[0] / 6 - x[1] >= 0

    def constraint2(self, x: np.ndarray) -> bool:
        return 7.5 - 0.5 * x[0] - x[1] >= 0
    
    def constraint3(self, x: np.ndarray) -> bool:
        return 30 - 5 * x[0] - x[1] >= 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mo = BinhKorn()
    optimizer = nevergrad.optimizers.DE(parametrization=mo.parametrization, budget=500)

    optimizer.tell(nevergrad.p.MultiobjectiveReference(), [10, 10])

    optimizer.minimize(mo, verbosity=2)

    # The function embeds its Pareto-front:
    print("Pareto front:")
    params = sorted(optimizer.pareto_front(), key=lambda p: p.losses[0])
    for param in params:
        print(f"{param} with losses {param.losses}")
    plt.scatter(x=[p.losses[0] for p in params], y=[p.losses[1] for p in params])
    plt.xlabel("f1(x)")
    plt.ylabel("f2(x)")
    plt.title(f"Pareto front for {mo.name}")
    plt.grid()
    plt.show()

    # It can also provide subsets:
    print("Random subset:", optimizer.pareto_front(2, subset="random"))
    print("Loss-covering subset:", optimizer.pareto_front(2, subset="loss-covering"))
    print("Domain-covering subset:", optimizer.pareto_front(2, subset="domain-covering"))
    print("EPS subset:", optimizer.pareto_front(2, subset="EPS"))

    # DOC_MULTIOBJ_OPT_1
    assert len(optimizer.pareto_front()) > 1
    assert len(optimizer.pareto_front(2, "loss-covering")) == 2
    assert len(optimizer.pareto_front(2, "domain-covering")) == 2
    assert len(optimizer.pareto_front(2, "hypervolume")) == 2
    assert len(optimizer.pareto_front(2, "random")) == 2
    assert len(optimizer.pareto_front(2, "EPS")) == 2
