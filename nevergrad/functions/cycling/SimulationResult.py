#!/usr/bin/env python3

import math

class SimulationResult:

	def __init__(self, finishTime, proportionCompleted, energyRemaining, velocityProfile):
		self.finishTime = finishTime
		self.energyRemaining = energyRemaining
		self.proportionCompleted = proportionCompleted
		self.velocityProfile = velocityProfile
		self.results = []

	def getFinishTime(self):
		return self.finishTime
	
	def getProportionCompleted(self):
		return self.proportionCompleted
	
	def getEnergyRemaining(self):
		return self.energyRemaining
	
	def getVelocityProfile(self):
		return self.velocityProfile
	
	def toString(self):
		output = "Simulation Result\n-----------------\n"
		if self.finishTime < math.inf:
			output = output + "Finish Time: " + str(self.finishTime) + " seconds\n"
			for i in range(0, len(self.energyRemaining)):
				output = output + "Cyclist " + str(i+1) + " Energy Remaining: " + str(self.energyRemaining[i]) + " joules\n"
		else:
			output = output + "Riders had insufficient energy for race completion\n" + "Proportion of race completed: " + str(self.proportionCompleted * 100) + "%\n"
		return output
