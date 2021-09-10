	#!/usr/bin/env python3

import math

class TTPSolution:


	def __init__(self, tspTour, packingPlan):
		self.tspTour = tspTour
		self.packingPlan = packingPlan

		self.fp = float("-inf")
		self.ft = float("inf")
		self.ftraw = math.inf
		self.ob = float("-inf")
		self.wend = float("inf")
		self.wendUsed = float("inf")
		self.computationTime = math.inf

	def reset(self):

		self.fp = float("-inf")
		self.ft = float("inf")
		self.ftraw = math.inf
		self.ob = float("-inf")

		self.wend = float("inf")
		self.wendUsed = float("inf")
		self.computationTime = math.inf

	def print(self):
		print(str(self.wend) + " " + str(self.wendUsed) + " " + str(self.fp) + " " + str(self.ftraw) + " " + str(self.ft) + " " + str(self.ob) + " " + str(self.computationTime))

	def printFull(self):
		self.print()
		print("tspTour " + str(self.tspTour).strip("[]"))
		print("packingPlan " + str(self.packingPlan).strip("[]"))

	def getObjective(self):
		return self.ob

