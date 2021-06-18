#!/usr/bin/env python3

from .TeamPursuit import TeamPursuit
from .Cyclist import Cyclist
from .SimulationResult import SimulationResult
import sys
import math

class WomensTeamPursuit(TeamPursuit):

	TEAM_SIZE = 3
	RACE_DISTANCE = 3000
	LAP_DISTANCE = 250
	RACE_SEGMENTS = int((2 * (RACE_DISTANCE / LAP_DISTANCE)) - 1)
	MAXIMUM_TRANSITIONS = RACE_SEGMENTS - 1

	def __init__(self):
		super().updateAirDensity()
		self.team = []
		for i in range(0, self.TEAM_SIZE):
			self.team.append(Cyclist(1.60, 65.0, 5.0, self, i+1, "female"))

	def simulate(self, transitionStrategy, pacingStrategy):
		
		if (len(transitionStrategy) != self.MAXIMUM_TRANSITIONS):
			sys.exit("Transition strategy for the womens team pursuit must have exactly " + str(self.MAXIMUM_TRANSITIONS) +" elements")
		if (len(pacingStrategy) != self.RACE_SEGMENTS):
			sys.exit("Pacing strategy for the womens team pursuit must have exactly " + str(self.RACE_SEGMENTS) + " elements")
		for i in range(0, self.RACE_SEGMENTS):
			if (pacingStrategy[i] > Cyclist.MAX_POWER or pacingStrategy[i] < Cyclist.MIN_POWER):
				sys.exit("All power elements of the pacing strategy must be in the range " + str(Cyclist.MIN_POWER) + "-" + str(Cyclist.MAX_POWER) + " Watts")
		
		for i in range(0, len(self.team)):
			self.team[i].reset()
		
		velocityProfile = [None] * self.RACE_SEGMENTS
		proportionCompleted = 0
		raceTime = 0
		for i in range(0, self.RACE_SEGMENTS):
			if (i == 0 or i == (self.RACE_SEGMENTS - 1)):
				distance = 187.5
			else:
				distance = 125.0
			if (super().cyclistsRemaining() == 3):
				
				if (i >= 1 and transitionStrategy[i-1]):
					super().transition()
					raceTime += TeamPursuit.TRANSITION_TIME

				leader = super().leader()
				time = 0.0;
				distanceRidden = 0.0;
				while (distanceRidden < distance):
					dist = leader.setPace(pacingStrategy[i])
					
					for j in range(0, len(self.team)):
						if (self.team[j].getPosition() > 1):
							self.team[j].follow(dist)
					
					if (distanceRidden + dist <= distance):
						distanceRidden += dist
					else:
						distanceRidden = distance
				
					time += self.TIME_STEP

				leader.increaseFatigue()
				for j in range(0, len(self.team)):
					if (self.team[j].getPosition() > 1):
						self.team[j].recover()
				
				if (super().cyclistsRemaining() >= 3):
					velocityProfile[i] = distance / time
					raceTime += time
					proportionCompleted += distance / self.RACE_DISTANCE
				else:
					raceTime = math.inf
			else:
				raceTime = math.inf
				break
				
		remainingEnergies = []
		for i in range(0, len(self.team)):
			remainingEnergies.append(self.team[i].getRemainingEnergy())

		return SimulationResult(raceTime, proportionCompleted, remainingEnergies, velocityProfile)
