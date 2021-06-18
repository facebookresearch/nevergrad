#!/usr/bin/env python3

from abc import ABC, abstractmethod
import math
import sys

class TeamPursuit(ABC):

	#constants
	FRICTION_COEFFICIENT = 0.0025
	DRAFTING_COEFFICIENTS = [0.75, 0.65, 0.55]
	GRAVITATIONAL_ACCELERATION = 9.80665
	TIME_STEP = 0.1

	TRANSITION_TIME = 0.12

	relativeHumidity = 0.5

	temperature = 20.0
	barometricPressure = 1013.25
		
	airDensity = None
	team = None
		

	def setTemperature(self, temperature):
		if temperature < 0.0 or temperature > 40:
			sys.exit("Temperature must be in range 0-40C")
		else:
			self.temperature = temperature
			self.updateAirDensity()

	def setBarometricPressure(self, barometricPressure):
		if barometricPressure < 800.0 or barometricPressure > 1200.0:
			sys.exit("Barometric pressure must be in the range 800-1200 hPa")
		else:
			self.barometricPressure = barometricPressure
			self.updateAirDensity()
	
	def setRelativeHumidity(self, relativeHumidity):
		if relativeHumidity < 0.0 or relativeHumidity > 1.0:
			sys.exit("Relative humidity must be in the range 0-1")
		else:
			self.relativeHumidity = relativeHumidity
			self.updateAirDensity()
	
	def setHeight(self, cyclistId, height):
		if cyclistId >= len(self.team):
			sys.exit("Cyclist identifier must be in the range 0-" + str(len(self.team)))
		else:
			self.team[cyclistId].setHeight(height)
	
	def setWeight(self, cyclistId, weight):
		if cyclistId >= len(self.team):
			sys.exit("Cyclist identifier must be in the range 0-" + str(len(self.team)))
		else:
			self.team[cyclistId].setWeight(weight)
	
	def setMeanMaximumPower(self, cyclistId, meanMaximumPower):
		if cyclistId >= len(self.team):
			sys.exit("Cyclist identifier must be in the range 0-" + str(len(self.team)))
		else:
			self.team[cyclistId].setMeanMaximumPower(meanMaximumPower)

	def getTemperature(self):
		return self.temperature

	def getBarometricPressure(self):
		return self.barometricPressure

	def getRelativeHumidity(self):
		return self.relativeHumidity

	def getHeight(self, cyclistId):
		if cyclistId >= len(self.team):
			sys.exit("Cyclist identifier must be in the range 0-" + str(len(self.team)))
		else:
			return self.team[cyclistId].getHeight()
	
	def getWeight(self, cyclistId):
		if cyclistId >= len(self.team):
			sys.exit("Cyclist identifier must be in the range 0-" + str(len(self.team)))
		else:
			return self.team[cyclistId].getWeight()
	
	def getMeanMaximumPower(self, cyclistId):
		if cyclistId >= len(self.team):
			sys.exit("Cyclist identifier must be in the range 0-" + str(len(self.team)))
		else:
			return self.team[cyclistId].getMeanMaximumPower()
	
	@abstractmethod
	def simulate(self, transitionStrategy, pacingStrategy):
		pass

	def updateAirDensity(self):
		ppWaterVapour = 100 * self.relativeHumidity * (6.1078 * math.pow(10, (((7.5 * (self.temperature + 273.15)) - 2048.625))/(self.temperature + 273.15 - 35.85)))
		ppDryAir = 100 * self.barometricPressure - ppWaterVapour
		self.airDensity = (ppDryAir/(287.058 * (self.temperature + 273.15))) + (ppWaterVapour/(461.495 * (self.temperature + 273.15)))
	
	def cyclistsRemaining(self):
		cyclistsRemaining = 0
		for i in range(0, len(self.team)):
			if self.team[i].getRemainingEnergy() > 0.0:
				cyclistsRemaining += 1
			else:
				self.team[i].setPosition(0)
		return cyclistsRemaining

	def leader(self):
		for i in range(0, len(self.team)):
			if (self.team[i].getPosition()) == 1:
				return self.team[i]
		return None
	
	def transition(self):
		for i in range(0, len(self.team)):
			if self.team[i].getPosition() > 0:
				if self.team[i].getPosition() == 1:
					self.team[i].setPosition(self.cyclistsRemaining())
				else:
					self.team[i].setPosition(self.team[i].getPosition() - 1)
