#!/usr/bin/env python3

from .TeamPursuit import TeamPursuit
import math
import sys

class Cyclist:

    #static class variables
    MAX_POWER = 1200
    MIN_POWER = 200
    DRAG_COEFFICIENT = 0.65
    MECHANICAL_EFFICIENCY = 0.977
    BIKE_MASS = 7.7
    fatigueLevel = 0

    def __init__(self, height, weight, meanMaximumPower, event, startPosition, gender):
        
        self.height = height
        self.weight = weight
        self.meanMaximumPower = meanMaximumPower
        self.event = event
        self.startPosition = startPosition
        self.position = startPosition
        self.gender = gender
        self.currentVelocity = 0.0
        self.fatigueLevel = 0

        self.updateCDA()
        self.updateTotalEnergy()
        self.remainingEnergy = self.totalEnergy


    def setPace(self, power):
        fatigueFactor = 1 - (0.01 * self.fatigueLevel)

        deltaKE = ((power * self.MECHANICAL_EFFICIENCY * fatigueFactor) - (self.coefficientDragArea * 0.5 * self.event.airDensity * math.pow(self.currentVelocity, 3)) - (TeamPursuit.FRICTION_COEFFICIENT * (self.weight + self.BIKE_MASS) * TeamPursuit.GRAVITATIONAL_ACCELERATION * self.currentVelocity)) * TeamPursuit.TIME_STEP

        newVelocity = math.pow(((2 * deltaKE / (self.weight + self.BIKE_MASS)) + math.pow(self.currentVelocity, 2)), 0.5)
        acceleration = newVelocity - self.currentVelocity
        distance = (self.currentVelocity * TeamPursuit.TIME_STEP) + (0.5 * acceleration * math.pow(TeamPursuit.TIME_STEP, 2))
        
        self.currentVelocity = newVelocity
        
        if self.remainingEnergy > power * TeamPursuit.TIME_STEP:
            self.remainingEnergy -= power * TeamPursuit.TIME_STEP
        else:
            self.remainingEnergy = 0.0
        
        return distance
    
    def follow(self, distance):
        fatigueFactor = 1 - (0.01 * self.fatigueLevel)
        
        acceleration = 2 * (distance - (self.currentVelocity * TeamPursuit.TIME_STEP)) / math.pow(TeamPursuit.TIME_STEP, 2)
        newVelocity = self.currentVelocity + (acceleration * TeamPursuit.TIME_STEP)
        deltaKE = 0.5 * (self.weight + self.BIKE_MASS) * (newVelocity - self.currentVelocity)
        power = ((self.coefficientDragArea * TeamPursuit.DRAFTING_COEFFICIENTS[self.position - 2] * 0.5 * self.event.airDensity * math.pow(self.currentVelocity, 3)) + (TeamPursuit.FRICTION_COEFFICIENT * (self.weight + self.BIKE_MASS) * TeamPursuit.GRAVITATIONAL_ACCELERATION * self.currentVelocity) + (deltaKE / TeamPursuit.TIME_STEP)) / (self.MECHANICAL_EFFICIENCY * fatigueFactor)
        
        self.currentVelocity = newVelocity
        
        
        if self.remainingEnergy > power * TeamPursuit.TIME_STEP:
            self.remainingEnergy -= power * TeamPursuit.TIME_STEP
        else:
            self.remainingEnergy = 0.0
    
    def getHeight(self):
        return self.height
    
    def getWeight(self):
        return self.weight
    
    def getMeanMaximumPower(self):
        return self.meanMaximumPower
    
    def getRemainingEnergy(self):
        return self.remainingEnergy
    
    def getPosition(self):
        return self.position
    
    def setWeight(self, weight):
        self.weight = weight
        self.updateCDA()
        self.updateTotalEnergy()
    
    def setHeight(self, height):
        self.height = height
        self.updateCDA()
    
    def setMeanMaximumPower(self, meanMaximumPower):
        self.meanMaximumPower = meanMaximumPower
        self.updateTotalEnergy()
    
    def setPosition(self, position):
        self.position = position

    def increaseFatigue(self):
        self.fatigueLevel += 2
    
    def recover(self):
        if (self.fatigueLevel > 0):
            self.fatigueLevel -= 1
    
    def reset(self):
        self.remainingEnergy = self.totalEnergy
        self.position = self.startPosition
        self.fatigueLevel = 0
        self.currentVelocity = 0
    
    def updateCDA(self):
        self.coefficientDragArea = self.DRAG_COEFFICIENT * ((0.0293 * math.pow(self.height, 0.725))*(math.pow(self.weight, 0.425)) + 0.0604)
    
    def updateTotalEnergy(self):
        if self.gender == "male":
            self.totalEnergy = self.meanMaximumPower * self.weight * 240
        else:
            self.totalEnergy = self.meanMaximumPower * self.weight * 210