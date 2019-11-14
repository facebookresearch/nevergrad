from . import core


def test_multiobjective_function():
    mfunc = core.MultiobjectiveFunction(lambda x: x, (100, 100))
    tuples = [(110, 110), (110, 90), (80, 80), (50, 50), (50, 50), (80, 80), (30, 60), (60, 30)]
    values = []
    for tup in tuples:
        values.append(mfunc(tup))
    expected = [-100, float('inf'), -400, -2500.0, -2500.0, -2530.0, -3300.0, -4100.0]  # TODO check
    assert values == expected, f"Expected {expected} but got {values}"
    front = [p[0][0][0] for p in mfunc.pareto_front]
    expected_front = [(110, 110), (80, 80), (50, 50), (30, 60), (60, 30)]  # TODO check
    assert front == expected_front, f"Expected {expected_front} but got {front}"
