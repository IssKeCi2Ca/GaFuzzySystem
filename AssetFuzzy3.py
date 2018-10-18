# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 20:39:25 2018

@author: Siew Yaw Hoong
"""

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import date
from datetime import time
from datetime import timedelta
import random
import io
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math

###############################################################################
### Fuzzy one-time setup 
###############################################################################

# Set up MACD as an antecedent
MACD = ctrl.Antecedent(np.arange(-300, 201, 1), 'MACD')
MACD['low'] = fuzz.trapmf(MACD.universe, [-300, -300, -30, -20])
MACD['med'] = fuzz.trapmf(MACD.universe, [-30, -20, 20, 30])
MACD['high'] = fuzz.trapmf(MACD.universe, [20, 30, 200, 200])

# Set up RSI as an antecedent
RSI = ctrl.Antecedent(np.arange(0, 101, 1), 'RSI')
RSI['low'] = fuzz.trapmf(RSI.universe, [0, 0, 25, 35])
RSI['med'] = fuzz.trapmf(RSI.universe, [25, 35, 65, 75])
RSI['high'] = fuzz.trapmf(RSI.universe, [65, 75, 100, 100])

# Set up Buy/Sell/Hold as a consequent
buysellCon = ctrl.Consequent(np.arange(0, 11, 1), 'buysellCon')
buysellCon['sell'] = fuzz.trapmf(buysellCon.universe, [0, 0, 2, 3])
buysellCon['hold'] = fuzz.trapmf(buysellCon.universe, [2, 3, 7, 8])
buysellCon['buy'] = fuzz.trapmf(buysellCon.universe, [7, 8, 10, 10])

#    MACD.view()
#    RSI.view()
#    buysellCon.view()
#____________________________________________________________________

# Set up Buy/Sell/Hold as an antecedent
buysellAnt = ctrl.Antecedent(np.arange(0, 11, 1), 'buysellAnt')
buysellAnt['sell'] = fuzz.trapmf(buysellAnt.universe, [0, 0, 2, 3])
buysellAnt['hold'] = fuzz.trapmf(buysellAnt.universe, [2, 3, 7, 8])
buysellAnt['buy'] = fuzz.trapmf(buysellAnt.universe, [7, 8, 10, 10])

# Set up Cash Balance as an antecedent
cash = ctrl.Antecedent(np.arange(0, 10000001, 1), 'cash')
cash['low'] = fuzz.trapmf(cash.universe, [0, 0, 2000000, 3000000])
cash['med'] = fuzz.trapmf(cash.universe, [2000000, 3000000, 7000000, 8000000])
cash['high'] = fuzz.trapmf(cash.universe, [7000000, 8000000, 10000000, 10000000])

# Set up Trading Volume as a consequent
volume = ctrl.Consequent(np.arange(0, 11, 1), 'volume')
volume['low'] = fuzz.trapmf(volume.universe, [0, 0, 2, 3])
volume['med'] = fuzz.trapmf(volume.universe, [2, 3, 7, 8])
volume['high'] = fuzz.trapmf(volume.universe, [7, 8, 10, 10])

#    buysellAnt.view()
#    cash.view()
#    volume.view()
#____________________________________________________________________

# Set up the Buy/Sell/Hold rules
rule01 = ctrl.Rule(MACD['low'] & RSI['low'], buysellCon['hold'])
rule02 = ctrl.Rule(MACD['low'] & RSI['med'], buysellCon['sell'])
rule03 = ctrl.Rule(MACD['low'] & RSI['high'], buysellCon['sell'])
rule04 = ctrl.Rule(MACD['med'] & RSI['low'], buysellCon['buy'])
rule05 = ctrl.Rule(MACD['med'] & RSI['med'], buysellCon['hold'])
rule06 = ctrl.Rule(MACD['med'] & RSI['high'], buysellCon['sell'])
rule07 = ctrl.Rule(MACD['high'] & RSI['low'], buysellCon['buy'])
rule08 = ctrl.Rule(MACD['high'] & RSI['med'], buysellCon['buy'])
rule09 = ctrl.Rule(MACD['high'] & RSI['high'], buysellCon['hold'])
#rule01.view()
#rule02.view()
#rule03.view()

# Set up control system for Buy/Sell/Hold
buysellCtrl = ctrl.ControlSystem([rule01, rule02, rule03, rule04, rule05, \
                                  rule06, rule07, rule08, rule09])
#    buysellCtrl.view()

# Set up the Trading Volume rules
rule10 = ctrl.Rule(buysellAnt['sell'] & cash['low'], volume['low'])
rule11 = ctrl.Rule(buysellAnt['sell'] & cash['med'], volume['low'])
rule12 = ctrl.Rule(buysellAnt['sell'] & cash['high'], volume['med'])
rule13 = ctrl.Rule(buysellAnt['hold'] & cash['low'], volume['low'])
rule14 = ctrl.Rule(buysellAnt['hold'] & cash['med'], volume['med'])
rule15 = ctrl.Rule(buysellAnt['hold'] & cash['high'], volume['high'])
rule16 = ctrl.Rule(buysellAnt['buy'] & cash['low'], volume['med'])
rule17 = ctrl.Rule(buysellAnt['buy'] & cash['med'], volume['high'])
rule18 = ctrl.Rule(buysellAnt['buy'] & cash['high'], volume['high'])

# Set up control system for Trading Volume
volumeCtrl = ctrl.ControlSystem([rule10, rule11, rule12, rule13, rule14, \
                                 rule15, rule16, rule17, rule18])
#    volumeCtrl.view()


###############################################################################
### Fuzzy calculation (to be called as a function)
###############################################################################
    
def fuzzy(MACDinput, RSIinput, cashinput):
    
    # Buy/Sell/Hold Defuzzification
    buysellSim = ctrl.ControlSystemSimulation(buysellCtrl)
    buysellSim.input['MACD'] = MACDinput
    buysellSim.input['RSI'] = RSIinput
    buysellSim.compute()
#    buysellCon.view(sim=buysellSim)
    
    print("Recommended Buy/Sell/Hold = ", buysellSim.output['buysellCon'])
    
    # Trading Volume Defuzzification
    volumeSim = ctrl.ControlSystemSimulation(volumeCtrl)
    volumeSim.input['buysellAnt'] = buysellSim.output['buysellCon']
    volumeSim.input['cash'] = cashinput
    volumeSim.compute()
#    volume.view(sim=volumeSim)
    
    vol = math.trunc(volumeSim.output['volume'])
    print("Recommended Trading Volume = ", vol)

    return vol


###############################################################################
### Generate 20 random individuals
###############################################################################

def random20():
    
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

    pop[['RSIperiod','MValue','NValue']] = pop[['RSIperiod','MValue','NValue']].astype(int)
    pop[['Fitness']] = pop[['Fitness']].astype(float)
    pop.dtypes
    
    fitnessBest = pop.nlargest(1, columns=['Fitness'])
    return fitnessBest


###############################################################################
### Read in master database 2011-2016 ###
###############################################################################

FCPO = pd.read_csv('FCPO_day.csv')
FCPO['Date'] = pd.to_datetime(FCPO['Date'], format = "%d/%m/%Y")
FCPO.set_index('Date', inplace=True)
FCPO.head(30)


# Partition the data to years

#for yearnow in range(2014, 2017, 1):
#    dfTrain = FCPO[str(yearnow-3)]
#    dfGA = FCPO[str(yearnow-2)]
#    dfTest = FCPO[str(yearnow-1)]
#    dfTrade = FCPO[str(yearnow)]
        

###############################################################################
### Fitness calculation function ###
###############################################################################
    
###  Temporary inputs for testing ###
    
pd.set_option('display.expand_frame_repr', False) 

ma = 'TMA' 
m= 26  
n = 12 
rsi = 14  

s = 9  # period for MACD signal line
cashbal = 10000000  
yearnow = 2011  

dfFit = FCPO[str(yearnow)]  # temporary df for calculating fitness
dfFit = dfFit.reset_index()
dfFit.dtypes
dfFit.head(30)

# dfFit.loc[ : ,'Close']

### Fitness function ###

def fitness(ma, m, n, rsi):    
    
    ### Calculate RSI ###
    
    dfFit['Change'] = dfFit.Close.rolling(2).apply(lambda x: x[-1]-x[0], raw=True)  
    # ie in a rolling window with 2 items, calc last item minus first item
    
    conditionsGain = [ dfFit['Change'] > 0, dfFit['Change'] <= 0 ] 
    choicesGain = [ dfFit['Change'], 0 ]
    dfFit['Gain'] = np.select(conditionsGain, choicesGain, default = 0)
    
    conditionsLoss = [ dfFit['Change'] < 0, dfFit['Change'] >= 0 ] 
    choicesLoss = [ abs(dfFit['Change']), 0 ]
    dfFit['Loss'] = np.select(conditionsLoss, choicesLoss, default = 0)

    dfFit['AveGain'] = 'NaN'
    dfFit['AveLoss'] = 'NaN'
    dfFit['AveGain'] = pd.to_numeric(dfFit['AveGain'], errors='coerce')
    dfFit['AveLoss'] = pd.to_numeric(dfFit['AveLoss'], errors='coerce')
    dfFit.loc[rsi, 'AveGain'] = dfFit.loc[1:14, 'Gain'].mean()  # initiate first AveGain
    dfFit.loc[rsi, 'AveLoss'] = dfFit.loc[1:14, 'Loss'].mean()  # initiate first AveLoss
    
    # fill in subsequent AveGain and AveLoss  # make this faster!
    for i in range(rsi+1, len(dfFit), 1):  
        dfFit.loc[i, 'AveGain'] = ( dfFit.loc[i-1, 'AveGain'] * (rsi-1) + dfFit.loc[i, 'Gain'] )/rsi
        dfFit.loc[i, 'AveLoss'] = ( dfFit.loc[i-1, 'AveLoss'] * (rsi-1) + dfFit.loc[i, 'Loss'] )/rsi
    
    dfFit['RS'] = dfFit['AveGain'] / dfFit['AveLoss']  
    dfFit['RSI'] = 'NaN'
    dfFit['RSI'] = pd.to_numeric(dfFit['RSI'], errors='coerce')
    
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
        
        dfFit['MAm'] = 'NaN'
        dfFit['MAn'] = 'NaN'
        dfFit['MAm'] = pd.to_numeric(dfFit['MAm'], errors='coerce')
        dfFit['MAn'] = pd.to_numeric(dfFit['MAn'], errors='coerce')
        
        dfFit.loc[m-1, 'MAm'] = dfFit.loc[:m-1, 'Close'].mean()  # initiate first AveGain
        dfFit.loc[m-1, 'MAn'] = dfFit.loc[n+2:m-1, 'Close'].mean()  # initiate first AveLoss
        
        # fill in subsequent AveGain and AveLoss  # make this faster!
        for i in range(m, len(dfFit), 1):  
            dfFit.loc[i, 'MAm'] = (dfFit.loc[i, 'Close'] - dfFit.loc[i-1, 'MAm']) * \
                mWeight + dfFit.loc[i-1, 'MAm']
        for i in range(m, len(dfFit), 1): 
            dfFit.loc[i, 'MAn'] = (dfFit.loc[i, 'Close'] - dfFit.loc[i-1, 'MAn']) * \
                nWeight + dfFit.loc[i-1, 'MAn']
                
        dfFit['EMA_MACD'] = dfFit['MAn'] - dfFit['MAm']  
        
        dfFit['Signal'] = 'NaN'  # initiate signal column
        dfFit['Signal'] = pd.to_numeric(dfFit['Signal'], errors='coerce')
        dfFit.loc[m+s-2, 'Signal'] = dfFit.loc[m-1:m+s-2, 'EMA_MACD'].mean()  # initiate first Signal

        # Fill in subsequent Signal
        for i in range(m+s-1, len(dfFit), 1):  
            dfFit.loc[i, 'Signal'] = (dfFit.loc[i, 'EMA_MACD'] - dfFit.loc[i-1, 'Signal']) * \
                sWeight + dfFit.loc[i-1, 'Signal']
                
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
    
    dfFit.dtypes
    dfFit.head(60)
    
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
    
    fitnessValue = dfFit['Asset'][len(dfFit)-1]
    print(fitnessValue)
    
    return fitnessValue
    
      
###############################################################################














