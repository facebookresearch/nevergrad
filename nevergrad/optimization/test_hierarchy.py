from . import hierarchy


def test_optimizer_hierarchy() -> None:
    optimizer_hierarchy = hierarchy.OptimizerHierarchy()
    optimizer_hierarchy.build_hierarchy()
