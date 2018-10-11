# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 22:25:33 2018
@author: Siew Yaw Hoong
SK Fuzzy Test
"""

#____________________________________________________________________

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

MACDinput = 80
RSIinput = 20
cashinput = 10000000

#____________________________________________________________________

# MACD.automf(3)
# RSI.automf(3)

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

# Set up Buy/Sell/Hold as a consequent
buysellCon = ctrl.Consequent(np.arange(0, 11, 1), 'buysellCon')
buysellCon['sell'] = fuzz.trapmf(buysellCon.universe, [0, 0, 2, 3])
buysellCon['hold'] = fuzz.trapmf(buysellCon.universe, [2, 3, 7, 8])
buysellCon['buy'] = fuzz.trapmf(buysellCon.universe, [7, 8, 10, 10])

MACD.view()
RSI.view()
buysellCon.view()

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

buysellAnt.view()
cash.view()
volume.view()

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
buysellCtrl.view()

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
volumeCtrl.view()

#____________________________________________________________________

# Buy/Sell/Hold Defuzzification
buysellSim = ctrl.ControlSystemSimulation(buysellCtrl)
buysellSim.input['MACD'] = MACDinput
buysellSim.input['RSI'] = RSIinput
buysellSim.compute()
buysellCon.view(sim=buysellSim)

print("Recommended Buy/Sell/Hold = ", buysellSim.output['buysellCon'])

# Trading Volume Defuzzification
volumeSim = ctrl.ControlSystemSimulation(volumeCtrl)
volumeSim.input['buysellAnt'] = buysellSim.output['buysellCon']
volumeSim.input['cash'] = cashinput
volumeSim.compute()
volume.view(sim=volumeSim)

print("Recommended Trading Volume = ", volumeSim.output['volume'])

#____________________________________________________________________
















