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
from datetime import datetime
from datetime import date
from datetime import time
from datetime import timedelta
import random
import io


###############################################################################
### Fitness calculation function ###
###############################################################################

def fitness(ma, m, n, rsi):
    
    
    ### Calculate MA ###
    
    if ma == 'SMA':
        dfFit['MAm'] = dfFit.Close.rolling(m).mean()
        dfFit['MAn'] = dfFit.Close.rolling(n).mean()  
        MAindex = m-1  # index to start the asset value calculation
            
#    elif ma == 'EMA':

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
    

    ### Calculate MACD

    mWeight = 2/(1+m)
    nWeight = 2/(1+n)
    sWeight = 2/(1+s)
    
    dfFit['EMAm'] = 'NaN'
    dfFit['EMAn'] = 'NaN'
    dfFit['EMAm'] = pd.to_numeric(dfFit['EMAm'], errors='coerce')
    dfFit['EMAn'] = pd.to_numeric(dfFit['EMAn'], errors='coerce')
    
    dfFit.loc[m-1, 'EMAm'] = dfFit.loc[:m-1, 'Close'].mean()  # initiate first EMAm
    dfFit.loc[m-1, 'EMAn'] = dfFit.loc[n+2:m-1, 'Close'].mean()  # initiate first EMAn
    
    # fill in subsequent EMAm and EMAn  # make this faster!
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
    
    # fill in subsequent AveGain and AveLoss  # make this faster!
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
        
    
    ### Calculate Asset Value ###
    
    maxindex = max(MAindex, MACDindex, RSIindex)
    
    for i in range(maxindex, len(dfFit), 1):      
        dfFit.loc[i, 'TradingVolRec'] = fuzzy(dfFit.loc[i, 'MAdelta'], dfFit.loc[i, 'MACDdelta'], dfFit.loc[i, 'RSI'])

    dfFit['TradingVolActual'] = 0    
    dfFit.loc[maxindex-1, 'Fee'] = 0.0        
    dfFit.loc[maxindex-1, 'ContractHeld'] = 0
    dfFit.loc[maxindex-1, 'CashBal'] = cashbal
    dfFit.loc[maxindex-1, 'Asset'] = cashbal
    
    buytrigger = 3
    selltrigger = -3

    # Daily Price is for 1 metric ton, so one Contract = 25 metric tonnes = 25 x Daily Price

    for i in range(maxindex, len(dfFit), 1):        
        
        if dfFit.loc[i, 'TradingVolRec'] > buytrigger:  # Buy recommendation

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

        elif dfFit.loc[i, 'TradingVolRec'] < selltrigger:  # Sell recommendation

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
       
    fitnessValue = dfFit['Asset'][len(dfFit)-1]

    return fitnessValue


###############################################################################
### Fuzzy one-time setup 
###############################################################################

# Set up RSI as an antecedent
MA = ctrl.Antecedent(np.arange(-200, 201, 1), 'MA')
MA['low'] = fuzz.trapmf(MA.universe, [-200, -200, -30, -20])
MA['med'] = fuzz.trapmf(MA.universe, [-30, -20, 20, 30])
MA['high'] = fuzz.trapmf(MA.universe, [20, 30, 200, 200])

# Set up MACD as an antecedent
MACD = ctrl.Antecedent(np.arange(-200, 201, 1), 'MACD')
MACD['low'] = fuzz.trapmf(MACD.universe, [-200, -200, -30, -20])
MACD['med'] = fuzz.trapmf(MACD.universe, [-30, -20, 20, 30])
MACD['high'] = fuzz.trapmf(MACD.universe, [20, 30, 200, 200])

# Set up RSI as an antecedent
RSI = ctrl.Antecedent(np.arange(0, 101, 1), 'RSI')
RSI['low'] = fuzz.trapmf(RSI.universe, [0, 0, 25, 35])
RSI['med'] = fuzz.trapmf(RSI.universe, [25, 35, 65, 75])
RSI['high'] = fuzz.trapmf(RSI.universe, [65, 75, 100, 100])

# Set up Trading Volume as a consequent
volume = ctrl.Consequent(np.arange(-10, 11, 1), 'volume')
volume['sell'] = fuzz.trapmf(volume.universe, [-10, -10, -7, -6])
volume['hold'] = fuzz.trapmf(volume.universe, [-7, -6, 6, 7])
volume['buy'] = fuzz.trapmf(volume.universe, [6, 7, 10, 10])

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
### Generate 20 random individuals
###############################################################################

# def random20():
    
#MAMethod = ['SMA', 'TPMA', 'TMA']
#RSIperiod = [5, 10, 14, 20, 25]
#MValue = [20, 50, 75, 100]
#NValue = [3, 5, 10, 15]    
#
#popCol = ['MAMethod', 'RSIperiod', 'MValue', 'NValue', 'Fitness']
#pop = pd.DataFrame(index=range(0,20,1), columns=popCol)
#
#for i in range(0, 20, 1):
#    pop.loc[i, 'MAMethod'] = random.choice(MAMethod)
#    pop.loc[i, 'RSIperiod'] = random.choice(RSIperiod)
#    pop.loc[i, 'MValue'] = random.choice(MValue)
#    pop.loc[i, 'NValue'] = random.choice(NValue)    
#    pop.loc[i, 'Fitness'] = fitness(pop.loc[i, 'MAMethod'], pop.loc[i, 'MValue'], pop.loc[i, 'NValue'], pop.loc[i, 'RSIperiod'])
#
#pop[['RSIperiod','MValue','NValue']] = pop[['RSIperiod','MValue','NValue']].astype(int)
#pop[['Fitness']] = pop[['Fitness']].astype(float)
#pop.dtypes
#
#fitnessBest = pop.nlargest(1, columns=['Fitness'])

#return fitnessBest


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
    
pd.set_option('display.expand_frame_repr', False) 

ma = 'SMA' 
m = 100
n = 10
rsi = 20

s = 9  # period for MACD signal line
cashbal = 10000000
maxfee = 2000  # estimate of maximum fee possible before buy/sell decision
  
yearnow = 2011  
dfFit = FCPO[str(yearnow)]  # temporary df for calculating fitness
dfFit = dfFit.reset_index()
dfFit.dtypes
dfFit.head(50)

f = fitness(ma, m, n, rsi)

print("")
print("For MA =", ma, ", m =", m, ", n =", n, ", RSI period =", rsi)
print("In year", yearnow)
print("Fitness value =", f)

dfFit.to_csv("dfFit_"+str(ma)+".csv")
      
###############################################################################














