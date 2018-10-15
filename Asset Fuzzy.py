# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 20:39:25 2018

@author: Siew Yaw Hoong
"""

#- partition data to years
#- calc sma etc as benchmark
#- output live trading 2014-16 to log table

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

#MACDinput = 100
#RSIinput = 20
#cashinput = 5000000

###############################################################################

# Fuzzy function.  Setup is in the Main function, it's static without looping;
# Processing is in a separate looping function

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
buysellCtrl = ctrl.ControlSystem([rule01, rule02, rule03, rule04, rule05, rule06, rule07, rule08, rule09])
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
volumeCtrl = ctrl.ControlSystem([rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18])
#    volumeCtrl.view()
#____________________________________________________________________
    
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

# Read in the initial database 2011-16.  Timenow 2014

FCPO = pd.read_csv('FCPO_day no Change.csv')
FCPO['Date'] = pd.to_datetime(FCPO['Date'], format = "%d/%m/%Y")
FCPO.set_index('Date', inplace=True)
FCPO.head(30)


# Partition the data to years

for i in range(2014, 2016, 1):
    FCPO[str(i)]

# First Training loop


###############################################################################

# Fitness calculation function

def fitness(ma, m, n, rsi):

    # Calculate RSI
    #    FCPO['RSI'] = np.random.randint(1, 99, FCPO.shape[0])
    
    FCPO['AveGain'+str(RSIperiod)] = FCPO.Gain.rolling(RSIperiod).mean()  # Fixed column
    FCPO['AveLoss'+str(RSIperiod)] = FCPO.Loss.rolling(RSIperiod).mean()  # Fixed column
    FCPO['RS'] = FCPO['AveGain'+str(RSIperiod)] / FCPO['AveLoss'+str(RSIperiod)]  # Changing column
    RSIformula = 100 - (100 / ( 1 + FCPO['RS'] ) )  
    FCPO['RSI'] = 0   # Changing column
    
    if FCPO['AveLoss'] != 0:
        FCPO['RSI'] = 100 - (100 / ( 1 + FCPO['RS'] ) ) 
    
    
    # Calculate MACD
    
    if ma = 'SMA':
                
        FCPO[ma+str(m)] = FCPO.Close.rolling(m).mean()  # Fixed column
        FCPO[ma+str(n)] = FCPO.Close.rolling(n).mean()  # Fixed column
        FCPO['MACD'] = FCPO[ma+str(n)] - FCPO[ma+str(m)]  # Changing column
            
    elif ma = 'EMA':
        do...
        
    elif ma = 'TMA':
        do..
        
    elif ma = 'TPMA':
        do..
        
    FCPO2 = FCPO.iloc[m-1:50]
    FCPO2.head(50)
    
    FCPO2['TradingVol'] = FCPO2.apply(lambda row: fuzzy(row['MACD'], row['RSI'], \
         cashbal), axis=1)
    
    FCPO2['ContractHeld'] = 0
    FCPO2['CashBal'] = cashbal
    FCPO2['Asset'] = cashbal


    for i in range(1, len(FCPO2)):
        
        # Buy contract
        if FCPO2['TradingVol'][i] > 0 and FCPO2['CashBal'][i] - 30 > FCPO2['TradingVol'][i] * FCPO2['High'][i]:
            FCPO2['ContractHeld'][i] = FCPO2['ContractHeld'][i-1] + FCPO2['TradingVol'][i]
            fee = max(30, 0.002 * FCPO2['TradingVol'][i] * FCPO2['High'][i])
            FCPO2['CashBal'][i]  = FCPO2['CashBal'][i-1]  - (FCPO2['TradingVol'][i] * FCPO2['High'][i]) - fee
            FCPO2['Asset'][i] = FCPO2['ContractHeld'][i] * FCPO2['Low'][i] + FCPO2['CashBal'][i] 
        
        # Sell contract
        elif FCPO2['TradingVol'][i] < 0 and FCPO2['ContractHeld'][i] > abs(FCPO2['TradingVol'][i]) and FCPO2['CashBal'][i]  > 30:
            FCPO2['ContractHeld'][i] = FCPO2['ContractHeld'][i-1] - abs(FCPO2['TradingVol'][i])
            fee = max(30, 0.002 * abs(FCPO2['TradingVol'][i]) * FCPO2['Low'])
            FCPO2['CashBal'][i]  = FCPO2['CashBal'][i-1]  + (FCPO2['TradingVol'][i] * FCPO2['Low'][i]) - fee
            FCPO2['Asset'][i] = FCPO2['ContractHeld'][i] * FCPO2['Low'][i] + FCPO2['CashBal'][i] 
    
    fitnessValue = FCPO2['Asset'][-1]
    
    return fitnessValue
    
      
###############################################################################














