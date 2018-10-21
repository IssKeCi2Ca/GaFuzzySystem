# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 20:39:25 2018

@author: Siew Yaw Hoong
"""

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import random
#from datetime import datetime
#from datetime import date
#from datetime import time
#from datetime import timedelta
#import io


###############################################################################
### Fitness calculation function ###
###############################################################################

def fitness(ma, m, n, rsi, yearstart, yearend):
    

    # Manually set df period for fitness calculation
    dfFit = FCPO[str(yearstart):str(yearend)]  # temporary df for calculating fitness
    dfFit = dfFit.reset_index()
#    dfFit.dtypes
#    dfFit.head(50)

    s = 9  # period for MACD signal line
    cashstart = 10000000
    maxfee = 100  # estimate of maximum fee possible before buy/sell decision
    buytrigger = 2
    selltrigger = -2
    
    ### Calculate MA ###
    
    if ma == 'SMA':
        dfFit['MAm'] = dfFit.Close.rolling(m).mean()
        dfFit['MAn'] = dfFit.Close.rolling(n).mean()  
        MAindex = m-1  # index to start the asset value calculation
            
    elif ma == 'EMA':
        mWeight = 2/(1+m)
        nWeight = 2/(1+n)
        
        dfFit['MAm'] = 'NaN'
        dfFit['MAn'] = 'NaN'
        dfFit['MAm'] = pd.to_numeric(dfFit['MAm'], errors='coerce')
        dfFit['MAn'] = pd.to_numeric(dfFit['MAn'], errors='coerce')
        
        dfFit.loc[m-1, 'MAm'] = dfFit.loc[:m-1, 'Close'].mean()  # initiate first MAm
        dfFit.loc[m-1, 'MAn'] = dfFit.loc[m-n:m-1, 'Close'].mean()  # initiate first MAn
        
        # fill in subsequent MAm and MAn  
        for i in range(m, len(dfFit), 1):  
            dfFit.loc[i, 'MAm'] = (dfFit.loc[i, 'Close'] - dfFit.loc[i-1, 'MAm']) * mWeight + dfFit.loc[i-1, 'MAm']
        for i in range(m, len(dfFit), 1): 
            dfFit.loc[i, 'MAn'] = (dfFit.loc[i, 'Close'] - dfFit.loc[i-1, 'MAn']) * nWeight + dfFit.loc[i-1, 'MAn']
                
        MAindex = m-1
    
    elif ma == 'TMA':
        dfFit['TMAm'] = dfFit.Close.rolling(m).mean()
        dfFit['TMAn'] = dfFit.Close.rolling(n).mean()  
        dfFit['MAm'] = dfFit.TMAm.rolling(m).mean()
        dfFit['MAn'] = dfFit.TMAn.rolling(n).mean()          
        MAindex = m*2-2
        
    elif ma == 'TPMA':
        dfFit['TPMA'] = dfFit.loc[:, ['High', 'Low', 'Close']].mean(axis=1)
        dfFit['MAm'] = dfFit.TPMA.rolling(m).mean()
        dfFit['MAn'] = dfFit.TPMA.rolling(n).mean()  
        MAindex = m-1
        
    dfFit['MAdelta'] = dfFit['MAn'] - dfFit['MAm']  
    

#    dfFit.dtypes
#    dfFit.head(150)
    
    ### Calculate MACD

    mWeight = 2/(1+m)
    nWeight = 2/(1+n)
    sWeight = 2/(1+s)
    
    dfFit['EMAm'] = 'NaN'
    dfFit['EMAn'] = 'NaN'
    dfFit['EMAm'] = pd.to_numeric(dfFit['EMAm'], errors='coerce')
    dfFit['EMAn'] = pd.to_numeric(dfFit['EMAn'], errors='coerce')
    
    dfFit.loc[m-1, 'EMAm'] = dfFit.loc[:m-1, 'Close'].mean()  # initiate first EMAm
    dfFit.loc[m-1, 'EMAn'] = dfFit.loc[m-n:m-1, 'Close'].mean()  # initiate first EMAn
    
    # fill in subsequent EMAm and EMAn 
    for i in range(m, len(dfFit), 1):  
        dfFit.loc[i, 'EMAm'] = (dfFit.loc[i, 'Close'] - dfFit.loc[i-1, 'EMAm']) * mWeight + dfFit.loc[i-1, 'EMAm']
    for i in range(m, len(dfFit), 1): 
        dfFit.loc[i, 'EMAn'] = (dfFit.loc[i, 'Close'] - dfFit.loc[i-1, 'EMAn']) * nWeight + dfFit.loc[i-1, 'EMAn']
            
    dfFit['MACD'] = dfFit['EMAn'] - dfFit['EMAm']  
    
    dfFit['Signal'] = 'NaN'  # initiate signal column
    dfFit['Signal'] = pd.to_numeric(dfFit['Signal'], errors='coerce')
    dfFit.loc[m+s-2, 'Signal'] = dfFit.loc[m-1:m+s-2, 'MACD'].mean()  # initiate first Signal

    # Fill in subsequent Signal
    for i in range(m+s-1, len(dfFit), 1):  
        dfFit.loc[i, 'Signal'] = (dfFit.loc[i, 'MACD'] - dfFit.loc[i-1, 'Signal']) * sWeight + dfFit.loc[i-1, 'Signal']
            
    dfFit['MACDdelta'] = dfFit['MACD'] - dfFit['Signal']  
    
    MACDindex = m+s-2
        
    
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
    
    # fill in subsequent AveGain and AveLoss  
    for i in range(rsi+1, len(dfFit), 1):  
        dfFit.loc[i, 'AveGain'] = ( dfFit.loc[i-1, 'AveGain'] * (rsi-1) + dfFit.loc[i, 'Gain'] ) / rsi
        dfFit.loc[i, 'AveLoss'] = ( dfFit.loc[i-1, 'AveLoss'] * (rsi-1) + dfFit.loc[i, 'Loss'] ) / rsi
    
    dfFit['RS'] = dfFit['AveGain'] / dfFit['AveLoss']  
    dfFit['RSI'] = 'NaN'
    conditionsRSI = [ dfFit['AveLoss'] != 0, dfFit['AveLoss'] == 0 ] 
    choicesRSI = [ 100 - (100 / ( 1 + dfFit['RS'] ) ) , 100 ]
    dfFit['RSI'] = np.select(conditionsRSI, choicesRSI)
    dfFit['RSI'] = pd.to_numeric(dfFit['RSI'], errors='coerce')

    RSIindex = rsi
        

#    dfFit.dtypes
#    dfFit.head(50)
    
    ### Calculate Asset Value ###
    
    maxindex = max(MAindex, MACDindex, RSIindex)
    
    for i in range(maxindex, len(dfFit), 1):      
        dfFit.loc[i, 'TradingVolRec'] = fuzzy(dfFit.loc[i, 'MAdelta'], dfFit.loc[i, 'MACDdelta'], dfFit.loc[i, 'RSI'])

    dfFit['TradingVolActual'] = 0    
    dfFit.loc[maxindex-1, 'Fee'] = 0.0        
    dfFit.loc[maxindex-1, 'ContractHeld'] = 0
    dfFit.loc[maxindex-1, 'CashBal'] = cashstart
    dfFit.loc[maxindex-1, 'Asset'] = cashstart
    
    # Daily Price is for 1 metric ton, so one Contract = 25 metric tonnes = 25 x Daily Price

    for i in range(maxindex, len(dfFit), 1):        
        
        if dfFit.loc[i, 'TradingVolRec'] >= buytrigger:  # Buy recommendation

            # Enough Cash to buy all the recommended volume
            if dfFit.loc[i-1, 'CashBal'] >= 25 * dfFit.loc[i, 'TradingVolRec'] * dfFit.loc[i, 'High'] + 2 * maxfee:
                dfFit.loc[i, 'Fee'] = max(30, 0.002 * dfFit.loc[i, 'TradingVolRec'] * dfFit.loc[i, 'High']) 
                dfFit.loc[i, 'TradingVolActual'] = dfFit.loc[i, 'TradingVolRec']
                dfFit.loc[i, 'ContractHeld'] = dfFit.loc[i-1, 'ContractHeld'] + dfFit.loc[i, 'TradingVolRec']
                dfFit.loc[i, 'CashBal'] = dfFit.loc[i-1, 'CashBal'] - dfFit.loc[i, 'Fee'] - (25 * dfFit.loc[i, 'TradingVolRec'] * dfFit.loc[i, 'High'])  
                dfFit.loc[i, 'Asset'] = 25 * dfFit.loc[i, 'ContractHeld'] * dfFit.loc[i, 'Low'] + dfFit.loc[i, 'CashBal']
 
            # Not enough Cash to buy all the recommended volume                           
            elif dfFit.loc[i-1, 'CashBal'] < 25 * dfFit.loc[i, 'TradingVolRec'] * dfFit.loc[i, 'High'] + 2 * maxfee:
                
                # Maximum number of Contract we can buy with existing Cash
                MaxTradingVol = math.trunc( (dfFit.loc[i-1, 'CashBal'] - 2 * maxfee) / dfFit.loc[i, 'High'] / 25)    
 
                if MaxTradingVol >= 1:                 
                    dfFit.loc[i, 'Fee'] = max(30, 0.002 * MaxTradingVol * dfFit.loc[i, 'High'])
                
                elif MaxTradingVol == 0:
                    dfFit.loc[i, 'Fee'] = 0.0

                dfFit.loc[i, 'TradingVolActual'] = MaxTradingVol
                dfFit.loc[i, 'ContractHeld'] = dfFit.loc[i-1, 'ContractHeld'] + MaxTradingVol             
                dfFit.loc[i, 'CashBal'] = dfFit.loc[i-1, 'CashBal'] - dfFit.loc[i, 'Fee'] - (25 * MaxTradingVol * dfFit.loc[i, 'High'])
                dfFit.loc[i, 'Asset'] = 25 * dfFit.loc[i, 'ContractHeld'] * dfFit.loc[i, 'Low'] + dfFit.loc[i, 'CashBal']

        elif dfFit.loc[i, 'TradingVolRec'] <= selltrigger:  # Sell recommendation

            # Enough ContractHeld to sell all the recommended volume            
            if dfFit.loc[i-1, 'ContractHeld'] >= abs(dfFit.loc[i, 'TradingVolRec']) and dfFit.loc[i-1, 'CashBal'] >= maxfee:
 
                dfFit.loc[i, 'Fee'] = max(30, 0.002 * abs(dfFit.loc[i, 'TradingVolRec']) * dfFit.loc[i, 'Low'])
                dfFit.loc[i, 'TradingVolActual'] = dfFit.loc[i, 'TradingVolRec']  # this would already be -ve
                dfFit.loc[i, 'ContractHeld'] = dfFit.loc[i-1, 'ContractHeld'] - abs(dfFit.loc[i, 'TradingVolRec'])
                dfFit.loc[i, 'CashBal']  = dfFit.loc[i-1, 'CashBal'] - dfFit.loc[i, 'Fee'] + (25 * abs(dfFit.loc[i, 'TradingVolActual']) * dfFit.loc[i, 'Low'])
                dfFit.loc[i, 'Asset'] = 25 * dfFit.loc[i, 'ContractHeld'] * dfFit.loc[i, 'Low'] + dfFit.loc[i, 'CashBal'] 

            # Not enough ContractHeld to sell all the recommended volume                      
            elif dfFit.loc[i-1, 'ContractHeld'] < abs(dfFit.loc[i, 'TradingVolRec']): 
                 
                if dfFit.loc[i-1, 'ContractHeld'] >= 1:                 
                    dfFit.loc[i, 'Fee'] = max(30, 0.002 * dfFit.loc[i-1, 'ContractHeld'] * dfFit.loc[i, 'Low'])
                    dfFit.loc[i, 'TradingVolActual'] = -(dfFit.loc[i-1, 'ContractHeld']) 
                    dfFit.loc[i, 'ContractHeld'] = dfFit.loc[i-1, 'ContractHeld'] - abs(dfFit.loc[i, 'TradingVolActual'])              

                elif dfFit.loc[i-1, 'ContractHeld'] == 0:
                    dfFit.loc[i, 'Fee'] = 0.0
                    dfFit.loc[i, 'TradingVolActual'] = 0     
                    dfFit.loc[i, 'ContractHeld'] = 0
                    
                dfFit.loc[i, 'CashBal']  = dfFit.loc[i-1, 'CashBal'] - dfFit.loc[i, 'Fee'] + (25 * abs(dfFit.loc[i, 'TradingVolActual']) * dfFit.loc[i, 'Low'])
                dfFit.loc[i, 'Asset'] = 25 * dfFit.loc[i, 'ContractHeld'] * dfFit.loc[i, 'Low'] + dfFit.loc[i, 'CashBal'] 

        else:  # Hold recommendation      
            dfFit.loc[i, 'Fee'] = 0.0 
            dfFit.loc[i, 'TradingVolActual'] = 0
            dfFit.loc[i, 'ContractHeld'] = dfFit.loc[i-1, 'ContractHeld']
            dfFit.loc[i, 'CashBal'] = dfFit.loc[i-1, 'CashBal']   
            dfFit.loc[i, 'Asset'] = 25 * dfFit.loc[i, 'ContractHeld'] * dfFit.loc[i, 'Low'] + dfFit.loc[i, 'CashBal']     
       
    fitnessValue = (dfFit['Asset'][len(dfFit)-1], yearstart, yearend, dfFit)
        
    return fitnessValue


###############################################################################
### Fuzzy one-time setup 
###############################################################################

# Set up RSI as an antecedent
MA = ctrl.Antecedent(np.arange(-1000, 401, 1), 'MA')
MA['low'] = fuzz.trapmf(MA.universe, [-1000, -400, -60, -40])
MA['med'] = fuzz.trapmf(MA.universe, [-60, -40, 40, 60])
MA['high'] = fuzz.trapmf(MA.universe, [40, 60, 400, 400])

# Set up MACD as an antecedent
MACD = ctrl.Antecedent(np.arange(-200, 201, 1), 'MACD')
MACD['low'] = fuzz.trapmf(MACD.universe, [-200, -200, -15, -8])
MACD['med'] = fuzz.trapmf(MACD.universe, [-15, -8, 8, 15])
MACD['high'] = fuzz.trapmf(MACD.universe, [8, 15, 200, 200])

# Set up RSI as an antecedent
RSI = ctrl.Antecedent(np.arange(0, 101, 1), 'RSI')
RSI['low'] = fuzz.trapmf(RSI.universe, [0, 0, 30, 40])
RSI['med'] = fuzz.trapmf(RSI.universe, [30, 40, 60, 70])
RSI['high'] = fuzz.trapmf(RSI.universe, [60, 70, 100, 100])

# Set up Trading Volume as a consequent
volume = ctrl.Consequent(np.arange(-10, 11, 1), 'volume')
volume['sell'] = fuzz.trapmf(volume.universe, [-10, -10, -6, -5])
volume['hold'] = fuzz.trapmf(volume.universe, [-6, -5, 5, 6])
volume['buy'] = fuzz.trapmf(volume.universe, [5, 6, 10, 10])

#____________________________________________________________________

# Set up the Buy/Sell/Hold rules
rule01 = ctrl.Rule(MA['low'] & MACD['low'] & RSI['low'], volume['sell'])
rule02 = ctrl.Rule(MA['low'] & MACD['low'] & RSI['med'], volume['sell'])
rule03 = ctrl.Rule(MA['low'] & MACD['low'] & RSI['high'], volume['sell'])
rule04 = ctrl.Rule(MA['low'] & MACD['med'] & RSI['low'], volume['hold'])
rule05 = ctrl.Rule(MA['low'] & MACD['med'] & RSI['med'], volume['sell'])
rule06 = ctrl.Rule(MA['low'] & MACD['med'] & RSI['high'], volume['sell'])
rule07 = ctrl.Rule(MA['low'] & MACD['high'] & RSI['low'], volume['buy'])
rule08 = ctrl.Rule(MA['low'] & MACD['high'] & RSI['med'], volume['hold'])
rule09 = ctrl.Rule(MA['low'] & MACD['high'] & RSI['high'], volume['sell'])

rule10 = ctrl.Rule(MA['med'] & MACD['low'] & RSI['low'], volume['hold'])
rule11 = ctrl.Rule(MA['med'] & MACD['low'] & RSI['med'], volume['sell'])
rule12 = ctrl.Rule(MA['med'] & MACD['low'] & RSI['high'], volume['sell'])
rule13 = ctrl.Rule(MA['med'] & MACD['med'] & RSI['low'], volume['buy'])
rule14 = ctrl.Rule(MA['med'] & MACD['med'] & RSI['med'], volume['hold'])
rule15 = ctrl.Rule(MA['med'] & MACD['med'] & RSI['high'], volume['sell'])
rule16 = ctrl.Rule(MA['med'] & MACD['high'] & RSI['low'], volume['buy'])
rule17 = ctrl.Rule(MA['med'] & MACD['high'] & RSI['med'], volume['buy'])
rule18 = ctrl.Rule(MA['med'] & MACD['high'] & RSI['high'], volume['hold'])

rule19 = ctrl.Rule(MA['high'] & MACD['low'] & RSI['low'], volume['buy'])
rule20 = ctrl.Rule(MA['high'] & MACD['low'] & RSI['med'], volume['hold'])
rule21 = ctrl.Rule(MA['high'] & MACD['low'] & RSI['high'], volume['sell'])
rule22 = ctrl.Rule(MA['high'] & MACD['med'] & RSI['low'], volume['buy'])
rule23 = ctrl.Rule(MA['high'] & MACD['med'] & RSI['med'], volume['buy'])
rule24 = ctrl.Rule(MA['high'] & MACD['med'] & RSI['high'], volume['hold'])
rule25 = ctrl.Rule(MA['high'] & MACD['high'] & RSI['low'], volume['buy'])
rule26 = ctrl.Rule(MA['high'] & MACD['high'] & RSI['med'], volume['buy'])
rule27 = ctrl.Rule(MA['high'] & MACD['high'] & RSI['high'], volume['buy'])
  
# Set up control system for Trading Volume
volumeCtrl = ctrl.ControlSystem([rule01, rule02, rule03, rule04, rule05, rule06, rule07, rule08, rule09,\
                                 rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18,\
                                 rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27])

    
###############################################################################
### Fuzzy calculation (to be called as a function)
###############################################################################
    
def fuzzy(MAinput, MACDinput, RSIinput):
    
    # Buy/Sell/Hold Defuzzification
    volumeSim = ctrl.ControlSystemSimulation(volumeCtrl)
    volumeSim.input['MA'] = MAinput
    volumeSim.input['MACD'] = MACDinput
    volumeSim.input['RSI'] = RSIinput
    volumeSim.compute()
    
    vol = math.trunc(volumeSim.output['volume'])
    print("Recommended Trading Volume = ", vol)
    
    return vol


###############################################################################
### Generate 20 random individuals & their Fitness Values
###############################################################################

# Use this to test random population
    
def random20():
    
    MAMethod = ['EMA', 'SMA', 'TPMA', 'TMA']
    MValue = [20, 50, 75, 100]
    NValue = [3, 5, 10, 15]    
    RSIperiod = [5, 10, 14, 20, 25]
    
    popCol = ['MAMethod', 'MValue', 'NValue', 'RSIperiod', 'Fitness']
    pop = pd.DataFrame(columns=popCol)
    #pop = pd.DataFrame(index=range(0,20,1), columns=popCol)
    
    for i in range(0, 20, 1):
        pop.loc[i, 'MAMethod'] = random.choice(MAMethod)
        pop.loc[i, 'MValue'] = random.choice(MValue)
        pop.loc[i, 'NValue'] = random.choice(NValue)   
        pop.loc[i, 'RSIperiod'] = random.choice(RSIperiod) 
        f = fitness(pop.loc[i, 'MAMethod'], pop.loc[i, 'MValue'], pop.loc[i, 'NValue'], pop.loc[i, 'RSIperiod'], yearstart, yearend)
        pop.loc[i, 'Fitness'] = f[0]
    
    pop[['RSIperiod','MValue','NValue']] = pop[['RSIperiod','MValue','NValue']].astype(int)
    pop[['Fitness']] = pop[['Fitness']].astype(float)
    
    fitnessBest = pop.nlargest(1, columns=['Fitness'])

    print("\n\n", pop)
    print("\nRandom individual with best fitness in year", f[1], "-", f[2])
    print("\n", fitnessBest)
    pop.to_csv("Random20.csv")
    
#    pop.dtypes

# return fitnessBest


###############################################################################
### Testing ###
###############################################################################

# Read in the master database
FCPO = pd.read_csv('FCPO_day.csv')
FCPO['Date'] = pd.to_datetime(FCPO['Date'], format = "%d/%m/%Y")
FCPO.set_index('Date', inplace=True)
#FCPO.head(30)
#pd.set_option('display.expand_frame_repr', False) 

# Specify the date range for df; do this first for either Test Option 1 or 2
yearstart = 2011
yearend = 2012


### Test Option 1 - Manually set inputs

ma = 'SMA' 
m = 50
n = 10
rsi = 25

f = fitness(ma, m, n, rsi, yearstart, yearend)
# Fitness function f to return:
# f[0] Fitness Value
# f[1] yearstart
# f[2] yearend
# f[3] dfFit

print("\nFor MA =", ma, ", m =", m, ", n =", n, ", RSI period =", rsi)
print("In years", f[1], "-", f[2])
print("Fitness value =", f[0])

f[3].to_csv("dfFit_"+str(yearstart)+"-"+str(yearend)+"_"+str(ma)+"_"+str(m)+"_"+str(n)+"_"+str(rsi)+".csv")


### Test Option 2 - Use random generator, uncomment & run command below only

#random20()


###############################################################################













