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

creator.create("FitnessMax", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

MAMethod = ['SMA', 'AMA', 'TFMA', 'TMA']
MNValues_MIN, MNValues_MAX = 1, 32
FuzzyExtent = ['EL', 'VL', 'L', 'M', 'H', 'VH', 'EH']
Recommend_MIN, Recommend_MAX = -10, 10

N_CYCLES = 1
POP_SIZE = 10
CXPB, MUTPB, NGEN= 0.5, 0.2, 20

toolbox.register("attr_mamethod", random.choice, MAMethod)
toolbox.register("attr_mnvalues", random.randint, MNValues_MIN, MNValues_MAX)
toolbox.register("attr_fuzzzyextent", random.choice, FuzzyExtent)
toolbox.register("attr_recommend", random.randint, Recommend_MIN, Recommend_MAX)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_mamethod, toolbox.attr_mnvalues, 
                  toolbox.attr_fuzzzyextent, toolbox.attr_recommend), n=N_CYCLES)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(POP_SIZE)

# printing the list using loop 
for x in range(POP_SIZE): 
   print (pop[x])

# def evaluateInd(individual):
#     # Do some computation
#     return result,

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
# toolbox.register("evaluate", evaluateInd)

for g in range(NGEN):
    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

# Apply crossover on the offspring
for child1, child2 in zip(offspring[::2], offspring[1::2]):
    if random.random() < CXPB:
        toolbox.mate(child1, child2)
        # del child1.fitness.values
        # del child2.fitness.values

# Apply mutation on the offspring
# for mutant in offspring:
#     if random.random() < MUTPB:
#         toolbox.mutate(mutant)
        # del mutant.fitness.values

# # Evaluate the individuals with an invalid fitness
# invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
# fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
# for ind, fit in zip(invalid_ind, fitnesses):
#     ind.fitness.values = fit

# The population is entirely replaced by the offspring
pop[:] = offspring

print('======================')
for x in range(POP_SIZE): 
   print (pop[x])
