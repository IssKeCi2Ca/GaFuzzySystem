# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 01:08:18 2018
@author: kooc
"""
# pylint: disable=no-member

import random
import numpy
import pprint
import pickle
import AssetFuzzy4
from pathlib import Path
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

class GA:

    def __init__(self):
        self.pop = [] 

    def selectPop(self, popSize):

        self.dfTrain=None
        self.cash_bal=None

        def evaluateInd(ind):
            # s = str(ind)
            # result = len(s)
            result = AssetFuzzy4.fitness(ind[0], ind[1], ind[2], ind[3], self.dfTrain, self.cash_bal)
            return result,

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()

        MAMethod = ['SMA', 'AMA', 'TFMA', 'TMA']
        MValue = ['10', '20', '50', '100', '200']
        NValue = ['1', '3', '5' ,'10', '15', '20']
        RSIPeriod = ['5','10', '14', '20', '25']

        N_CYCLES = 1

        toolbox.register("attr_mamethod", random.choice, MAMethod)
        toolbox.register("attr_mvalue", random.choice, MValue)
        toolbox.register("attr_nvalue", random.choice, NValue)
        toolbox.register("attr_rsiperiod", random.choice, RSIPeriod)

        toolbox.register("individual", tools.initCycle, creator.Individual,
                        (toolbox.attr_mamethod, toolbox.attr_mvalue, 
                        toolbox.attr_nvalue, toolbox.attr_rsiperiod), n=N_CYCLES)

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Operator registering
        toolbox.register("evaluate", evaluateInd)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=0, up=1, indpb=0.85)
        toolbox.register("select", tools.selBest)

        MU, LAMBDA = popSize, 20
        pop = toolbox.population(n=MU)
        hof = tools.HallOfFame(5)

        hof_file = Path("hof.p")
        if hof_file.is_file():
            ppop = pickle.load( open("hof.p", "rb"))
            hof.update(ppop)
            print('\n%d elems in the History' % len(ppop))
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(ppop)
           
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean, axis=0)
        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)

        pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                                    cxpb=0.7, mutpb=0.3, ngen=40, 
                                                    stats=stats, halloffame=hof, verbose=True)

        print('\n%d elems in the HallOfFame' % len(hof))
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(pop)

        pickle.dump(pop, open("hof.p", "wb"))
        return pop
	
    def CheckTrade(self, pd_rates):
        ind = self.pop[0]
        result_trade = AssetFuzzy4.fitness(ind[0], ind[1], ind[2], ind[3], self.pd_rates, self.cash_bal)
        return result_trade