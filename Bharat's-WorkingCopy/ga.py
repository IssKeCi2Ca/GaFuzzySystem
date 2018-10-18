# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 01:08:18 2018

@author: kooc
"""
# pylint: disable=no-member

import random
import numpy
import pprint
import AssetFuzzy2

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

class GA:

    def __init__(self):
        self.pop = []

    #################################Not Required####################################
    def Generate_Random_Individuals(popSize):
        # Generate 20 random individuals
        MAMethod = ['SMA', 'EMA', 'TFMA', 'TMA']
        RSIperiod = [5, 10, 14, 20, 25]
        MValue = [25, 50, 100, 150, 200]
        NValue = [3, 5, 10, 15, 20]    

        popCol = ['MAMethod', 'RSIperiod', 'MValue', 'NValue', 'Fitness']
        pop = pd.DataFrame(index=range(0,20,1), columns=popCol)

        for i in range(0, 20, 1):
            pop['MAMethod'][i] = random.choice(MAMethod)
            pop['RSIperiod'][i] = random.choice(RSIperiod)
            pop['MValue'][i] = random.choice(MValue)
            pop['NValue'][i] = random.choice(NValue)    
            pop['Fitness'][i] = fitness(pop['MAMethod'][i], pop['RSIperiod'][i], \
               pop['MValue'][i], pop['NValue'][i])

        pop

    ###############################################################################
    #define the GA Parameters - Pop Size and all other parameters passed from main  function to fuzzy
    def GAEvolve(popSize,ma, m, n, rsi,dfFit,s,cashbal,yearnow):

        ###############################################################################
        def evaluateInd(individual):
            # To test ga without using AssetFuzzy, uncomment the 2 lines below and comment the 3rd and 4th line below
            # s = str(individual)
            # result = len(s)
            result = self.fitness(ma, m, n, rsi,dfFit,s,cashbal,yearnow)
            #Fitness function calls fuzzy
            return result,
        ###############################################################################
        
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
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0)
        toolbox.register("select", tools.selBest)

        history = tools.History()
        # Decorate the variation operators
        toolbox.decorate("mate", history.decorator)
        toolbox.decorate("mutate", history.decorator)
        MU, LAMBDA = popSize, 20
        pop = toolbox.population(n=MU)
        history.update(pop)
        print('\n%d elems in the History' % len(pop))
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(pop)

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
        ##Best set of rules should be returned by fitness function and needs to be incorporated to next generation
        ##chekit / bharat to do

        print('\n%d elems in the HallOfFame' % len(hof))
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(pop)

        return pop
    ###############################################################################
    
    # Fitness calculation function
    #s- Period ID , dfFit - Period Dataset
    def fitness(ma, m, n, rsi,dfFit,s,cashbal,yearnow):    
        dfFit = dfFit.reset_index()
        dfFit.dtypes
        dfFit.head(60)
        ### Calculate RSI ###
        
        dfFit['Change'] = dfFit.Close.rolling(2).apply(lambda x: x[-1]-x[0])  
        # ie in a rolling window with 2 items, calc last item minus first item
        
        conditionsGain = [ dfFit['Change'] > 0, dfFit['Change'] <= 0 ] 
        choicesGain = [ dfFit['Change'], 0 ]
        dfFit['Gain'] = np.select(conditionsGain, choicesGain, default = 0)
        
        conditionsLoss = [ dfFit['Change'] < 0, dfFit['Change'] >= 0 ] 
        choicesLoss = [ abs(dfFit['Change']), 0 ]
        dfFit['Loss'] = np.select(conditionsLoss, choicesLoss, default = 0)
            
        dfFit['AveGain'] = 'NaN'
        dfFit['AveLoss'] = 'NaN'
        dfFit['AveGain'][rsi] = dfFit['Gain'].iloc[1:15].mean()  # initiate first AveGain
        dfFit['AveLoss'][rsi] = dfFit['Loss'].iloc[1:15].mean()  # initiate first AveLoss
        dfFit['AveGain'] = pd.to_numeric(dfFit['AveGain'], errors='coerce')
        dfFit['AveLoss'] = pd.to_numeric(dfFit['AveLoss'], errors='coerce')
        
        # fill in subsequent AveGain and AveLoss  # make this faster!
        for i in range(rsi+1, len(dfFit), 1):  
            dfFit['AveGain'][i] = ( dfFit['AveGain'][i-1] * (rsi-1) + dfFit['Gain'][i] )/rsi
            dfFit['AveLoss'][i] = ( dfFit['AveLoss'][i-1] * (rsi-1) + dfFit['Loss'][i] )/rsi
        
        dfFit['RS'] = dfFit['AveGain'] / dfFit['AveLoss']  
        dfFit['RSI'] = 0.0   
        
        conditionsRSI = [ dfFit['AveLoss'] != 0, dfFit['AveLoss'] == 0 ] 
        choicesRSI = [ 100 - (100 / ( 1 + dfFit['RS'] ) ) , 100 ]
        dfFit['RSI'] = np.select(conditionsRSI, choicesRSI)
            
        
        ### Calculate MACD ###
        
        if ma == 'SMA':
            dfFit['MAm'] = dfFit.Close.rolling(m).mean()
            dfFit['MAn'] = dfFit.Close.rolling(n).mean()  
            dfFit['MACD'] = dfFit['MAn'] - dfFit['MAm']  
                
        elif ma == 'EMA':
            mWeight = 2/(1+m)
            nWeight = 2/(1+n)
            sWeight = 2/(1+s)
            
            dfFit['MAm'] = 0.0
            dfFit['MAn'] = 0.0
            dfFit['MAm'][m] = dfFit['Close'].iloc[1:m].mean()  # initiate first AveGain
            
            # fill in subsequent AveGain and AveLoss  # make this faster!
            for i in range(m+1, len(dfFit), 1):  
                dfFit['MAm'][i] = (dfFit['Close'][i] - dfFit['MAm'][i-1]) * \
                    mWeight + dfFit['MAm'][i-1]
                dfFit['MAn'][i] = (dfFit['Close'][i] - dfFit['MAm'][i-1]) * \
                    nWeight + dfFit['MAm'][i-1]
                    
            dfFit['EMA_MACD'] = dfFit['MAn'] - dfFit['MAm']  
            dfFit['Signal'] = 0.0
            dfFit['Signal'][m+s] = dfFit['EMA_MACD'].iloc[1:s].mean()  # initiate first Signal

            # Fill in subsequent Signal
            for i in range(m+s+1, len(dfFit), 1):  
                dfFit['Signal'][i] = (dfFit['EMA_MACD'][i] - dfFit['Signal'][i-1]) * \
                    sWeight + dfFit['Signal'][i-1]
                    
            dfFit['MACD'] = dfFit['EMA_MACD'] - dfFit['Signal']  

        elif ma == 'TMA':
            dfFit['SMAm'] = dfFit.Close.rolling(m).mean()
            dfFit['SMAn'] = dfFit.Close.rolling(n).mean()  
            dfFit['MAm'] = dfFit.SMAm.rolling(m).mean()
            dfFit['MAn'] = dfFit.SMAn.rolling(n).mean()          
            dfFit['MACD'] = dfFit['MAn'] - dfFit['MAm']  
            
        elif ma == 'TPMA':
            dfFit['TPMA'] = dfFit.loc[:, ['High', 'Low', 'Close']].mean(axis=1)
            dfFit['MAm'] = dfFit.TPMA.rolling(m).mean()
            dfFit['MAn'] = dfFit.TPMA.rolling(n).mean()  
            dfFit['MACD'] = dfFit['MAn'] - dfFit['MAm']  
            
            
        ### Calculate Asset Value ###
        
        dfFit.head(50)
        # use fuzzy logic 
        dfFit['TradingVol'] = dfFit.apply(lambda row: fuzzy(row['MACD'], \
              row['RSI'], cashbal), axis=1)
        
        dfFit['ContractHeld'] = 0
        dfFit['CashBal'] = cashbal
        dfFit['Asset'] = cashbal


        for i in range(1, len(dfFit)):
            
            # Buy contract
            if dfFit['TradingVol'][i] > 0 and dfFit['CashBal'][i] - 30 > dfFit['TradingVol'][i] * dfFit['High'][i]:
                dfFit['ContractHeld'][i] = dfFit['ContractHeld'][i-1] + dfFit['TradingVol'][i]
                fee = max(30, 0.002 * dfFit['TradingVol'][i] * dfFit['High'][i])
                dfFit['CashBal'][i]  = dfFit['CashBal'][i-1]  - (dfFit['TradingVol'][i] * dfFit['High'][i]) - fee
                dfFit['Asset'][i] = dfFit['ContractHeld'][i] * dfFit['Low'][i] + dfFit['CashBal'][i] 
            
            # Sell contract
            elif dfFit['TradingVol'][i] < 0 and dfFit['ContractHeld'][i] > abs(dfFit['TradingVol'][i]) and dfFit['CashBal'][i]  > 30:
                dfFit['ContractHeld'][i] = dfFit['ContractHeld'][i-1] - abs(dfFit['TradingVol'][i])
                fee = max(30, 0.002 * abs(dfFit['TradingVol'][i]) * dfFit['Low'])
                dfFit['CashBal'][i]  = dfFit['CashBal'][i-1]  + (dfFit['TradingVol'][i] * dfFit['Low'][i]) - fee
                dfFit['Asset'][i] = dfFit['ContractHeld'][i] * dfFit['Low'][i] + dfFit['CashBal'][i] 
        
        fitnessValue = dfFit['Asset'][-1]
        
        return fitnessValue
        ###############################################################################
