import numpy as np
from . import core

def test_PBO() -> None:
    for fid in range(1,24):
        func = core.PBOFunction(fid, 0, 16)
        x = np.random.randint(0,2,(func.dimension,))
        value = func(x)  

def test_W_model() -> None:
    func = core.WModelFunction()
    x = np.random.randint(0,2,(func.dimension,))
    func2 = core.PBOFunction(1, 0, 16)
    assert func(x) == func2(x), "W-model with default setting should equal base_function"
    func = core.WModelFunction(base_function = "LeadingOnes")
    x = np.random.randint(0,2,(func.dimension,))
    func2 = core.PBOFunction(2, 0, 16)
    assert func(x) == func2(x), "W-model with default setting should equal base_function"
    