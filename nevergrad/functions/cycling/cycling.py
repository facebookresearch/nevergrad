# Creator: Ryan Kroon, Email: rkroon19@gmail.com
# Accompanying paper: Extending Nevergrad, an Optimisation Platform (in directory)

# Evolving Pacing Strategies for Team Pursuit Track Cycling
# Markus Wagner, Jareth Day, Diora Jordan, Trent Kroeger, Frank Neumann
# Advances in Metaheuristics, Vol. 53, Springer, 2013.
# https://link.springer.com/chapter/10.1007/978-1-4614-6322-1_4
# Java code: https://cs.adelaide.edu.au/~markus/pub/TeamPursuit.zip

from .WomensTeamPursuit import WomensTeamPursuit
from .MensTeamPursuit import MensTeamPursuit
import random
import numpy as np
import nevergrad as ng
from nevergrad.parametrization import parameter as p
from .. import base

class Cycling(base.ExperimentFunction):
	"""
	Team Pursuit Track Cycling Simulator.

	Parameters
	----------
	Strategy: int
		Refers to Transition strategy or Pacing Strategy (or both) of the cyclists.
	"""

	def __init__(self, Strategy: int = 30) -> None:

		# optimising transition strategy for men's team
		if Strategy == 30:
			strategy = p.Choice([False, True], repetitions=Strategy)
			parameter = p.Instrumentation(strategy).set_name("")
			super().__init__(MensTeamPursuitSimulation, parameter)

		# optimising pacing strategy for men's team
		elif Strategy == 31:
			init = 550 * np.ones(Strategy)
			parameter = p.Array(init=init)
			parameter.set_bounds(200, 1200)
			parameter.set_name("Mens Pacing Strategy")
			super().__init__(MensTeamPursuitSimulation, parameter)

		# optimising pacing and transition strategies for men's team
		elif Strategy == 61:
			init = 0.5 * np.ones(Strategy)
			parameter = p.Array(init=init)
			parameter.set_bounds(0, 1)
			parameter.set_name("Pacing and Transition")
			super().__init__(MensTeamPursuitSimulation, parameter)

		# optimising transition strategy for women's team
		elif Strategy == 22:
			strategy = ng.p.Choice([False, True], repetitions=Strategy)
			parameter = ng.p.Instrumentation(strategy).set_name("")
			super().__init__(WomensTeamPursuitSimulation, parameter)

		# optimising pacing strategy for women's team
		elif Strategy == 23:
			init = 400 * np.ones(Strategy)
			parameter = p.Array(init=init)
			parameter.set_bounds(200, 1200)
			parameter.set_name("Womens Pacing Strategy")
			super().__init__(WomensTeamPursuitSimulation, parameter)

		# optimising pacing and transition strategies for women's team
		elif Strategy == 45:
			init = 0.5 * np.ones(Strategy)
			parameter = p.Array(init=init)
			parameter.set_bounds(0, 1)
			parameter.set_name("Pacing and Transition")
			super().__init__(WomensTeamPursuitSimulation, parameter)


def MensTeamPursuitSimulation(x: np.ndarray) -> float:

	if len(x) == 30:
		MENS_TRANSITION_STRATEGY = x
		MENS_PACING_STRATEGY = [550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550, 550]

	elif len(x) == 31:
		MENS_TRANSITION_STRATEGY = [True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False]
		MENS_PACING_STRATEGY = x

	elif len(x) == 45:
		MENS_TRANSITION_STRATEGY = x[:30]
		for i in range (0, len(MENS_TRANSITION_STRATEGY)):
			if MENS_TRANSITION_STRATEGY[i] < 0.5:
				MENS_TRANSITION_STRATEGY[i] = False
			elif MENS_TRANSITION_STRATEGY[i] > 0.5:
				MENS_TRANSITION_STRATEGY[i] = True
			elif MENS_TRANSITION_STRATEGY[i] == 0.5:
				MENS_TRANSITION_STRATEGY[i] = random.choice([True, False])

		MENS_PACING_STRATEGY = x[30:]
		for i in range  (0, len(MENS_PACING_STRATEGY)):
			MENS_PACING_STRATEGY[i] = 100 * MENS_PACING_STRATEGY[i] + 200
	
	# Create a MensTeamPursuit object
	mensTeamPursuit = MensTeamPursuit()
	
	# Simulate event with the default strategies
	result = mensTeamPursuit.simulate(MENS_TRANSITION_STRATEGY, MENS_PACING_STRATEGY)

	#print(result.getFinishTime())

	if result.getFinishTime() > 10000: # in case of inf
		return 10000
	else:
		return float(result.getFinishTime())

def WomensTeamPursuitSimulation(x: np.ndarray) -> float:

	if len(x) == 22:
		WOMENS_TRANSITION_STRATEGY = x
		WOMENS_PACING_STRATEGY = [400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400]

	elif len(x) == 23:
		WOMENS_TRANSITION_STRATEGY = [True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False]
		WOMENS_PACING_STRATEGY = x

	elif len(x) == 45:
		WOMENS_TRANSITION_STRATEGY = x[:22]
		for i in range (0, len(WOMENS_TRANSITION_STRATEGY)):
			if WOMENS_TRANSITION_STRATEGY[i] < 0.5:
				WOMENS_TRANSITION_STRATEGY[i] = False
			elif WOMENS_TRANSITION_STRATEGY[i] > 0.5:
				WOMENS_TRANSITION_STRATEGY[i] = True
			elif WOMENS_TRANSITION_STRATEGY[i] == 0.5:
				WOMENS_TRANSITION_STRATEGY[i] = random.choice([True, False])

		WOMENS_PACING_STRATEGY = x[22:]
		for i in range  (0, len(WOMENS_PACING_STRATEGY)):
			WOMENS_PACING_STRATEGY[i] = 100 * WOMENS_PACING_STRATEGY[i] + 200

	# Create a WomensTeamPursuit object
	womensTeamPursuit = WomensTeamPursuit()
			
	# Simulate event with the default strategies
	result = womensTeamPursuit.simulate(WOMENS_TRANSITION_STRATEGY, WOMENS_PACING_STRATEGY)

	#print(result.getFinishTime())

	if result.getFinishTime() > 10000: # in case of inf
		return 10000
	else:
		return float(result.getFinishTime())
