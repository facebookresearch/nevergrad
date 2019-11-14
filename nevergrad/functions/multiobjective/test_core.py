from . import core

def test_multiobjective_function():
    mfunc = core.MultiobjectiveFunction(lambda x: x, (100, 100))
    tuples = [(110, 110),     # -0 + 
              (110, 90),      # -0 +
              (80, 80),       # -400
              (50, 50),       # -2500
              (50, 50),       # -2500
              (80, 80),       # -2470
              (30, 60),       # [30,50]x[60,100] + [50,100]x[50,100] --> -2500 -800 = -3300 
              (60, 30)]       # [30,50]x[60,100] + [50,100]x[50,100] + [60,100]x[30,50] --> -2500 -800 -800= -4100 
    values = []
    for tup in tuples:
        values.append(mfunc(tup))
    print("values=", values)
    expected = [-100, float('inf'), -400, -2500.0, -2500.0, -2530.0, -3300.0, -4100.0]  # TODO check
    assert values == expected, f"Expected {expected} but got {values}"
    front = [p[0][0][0] for p in mfunc.pareto_front]
    print("font=", front)
    expected_front = [(110, 110),(80, 80), (50, 50), (30, 60), (60, 30)]  # TODO check
    assert front == expected_front, f"Expected {expected_front} but got {front}"
