# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 01:08:18 2018

@author: kooc
"""
# pylint: disable=no-member

import random

from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

MAMethod = ['SMA', 'AMA', 'TFMA', 'TMA']
MNValues_MIN, MNValues_MAX = 1, 32
FuzzyExtent = ['EL', 'VL', 'L', 'M', 'H', 'VH', 'EH']
Recommend_MIN, Recommend_MAX = -10, 10

N_CYCLES = 10
POP_SIZE = 20

toolbox.register("attr_mamethod", random.choice, MAMethod)
toolbox.register("attr_mnvalues", random.randint, MNValues_MIN, MNValues_MAX)
toolbox.register("attr_fuzzzyextent", random.choice, FuzzyExtent)
toolbox.register("attr_recommend", random.randint, Recommend_MIN, Recommend_MAX)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_mamethod, toolbox.attr_mnvalues, 
                  toolbox.attr_fuzzzyextent, toolbox.attr_recommend), n=N_CYCLES)

# ind = toolbox.individual()
# print(ind)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(POP_SIZE)

# printing the list using loop 
for x in range(POP_SIZE): 
   print (pop[x])