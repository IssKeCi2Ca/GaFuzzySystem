# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 01:08:18 2018

@author: kooc
"""
# pylint: disable=no-member

import random
import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

class GA:

    def __init__(self):
        self.pop = []

    def selectPop(popSize):

        def evaluateInd(individual):
            #Get from Fuzzy?    
            s = str(individual)
            result = len(s)
            return result,

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()

        MAMethod = ['SMA', 'AMA', 'TFMA', 'TMA']
        RSIPeriod = ['5','10', '14', '20', '25']
        MValue = ['10', '20', '50', '100', '200']
        NValue = ['1', '3', '5' ,'10', '15', '20']

        N_CYCLES = 1

        toolbox.register("attr_mamethod", random.choice, MAMethod)
        toolbox.register("attr_rsiperiod", random.choice, RSIPeriod)
        toolbox.register("attr_mvalue", random.choice, MValue)
        toolbox.register("attr_nvalue", random.choice, NValue)

        toolbox.register("individual", tools.initCycle, creator.Individual,
                        (toolbox.attr_mamethod, toolbox.attr_rsiperiod, 
                        toolbox.attr_mvalue, toolbox.attr_nvalue), n=N_CYCLES)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Operator registering
        toolbox.register("evaluate", evaluateInd)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0)
        toolbox.register("select", tools.selBest)

        MU, LAMBDA = popSize, 20
        pop = toolbox.population(n=MU)
        # hof = tools.ParetoFront()
        hof = tools.HallOfFame(10)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean, axis=0)
        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)

        pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                                    cxpb=0.7, mutpb=0.3, ngen=40, 
                                                    stats=stats, halloffame=hof)

        print('\n%d elems in the HallOfFame' % len(hof))
        print (hof)
        
        return pop
