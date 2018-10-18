import numpy as np #matrix math
# written in a separate file ga
import ga
#Fuzzy Logic Code Goes in here
import AssetFuzzy2
import pandas as pd
from datetime import datetime
from datetime import date
from datetime import time
from datetime import timedelta

#Implementation of Trading System - Updating Stocks,Cash Balance
class Trading:
        #Program divided into 4 parts.
        #1) Initialise the required variables
        #2) Define Helper Functions such as Update Date Time, Audit Date Changes, Download Audit Report
        #3) Implement GA & Fuzzy logic in Training and Trading Methods
        #4) Initialise the required variables
################################################################# 1. Initialise the required dataframes and variables ##########################################################        
	def __init__(self):
                #initialise other variables SystemDate, Trading Reqd
		initMarketRates()
		#load our GA and Fuzzy Models
		Training_Done= 0
		Trading_Required=0
		System_Date = datetime.datetime(2011, 1, 1, 00, 00)
		#year, month, date, hh, mm
		popSize = 10
		i=0
		#pop = ga.GA.Generate_Random_Individuals(popSize)

################################################################# 2.Helper Functions (Start) ###################################################################################

        #Load the Rates in a Pandas Frame. All the Data Given can be converted to a CSV And imported into a frame
	def initMarketRates():
                #All the Data Given can be converted to a CSV And imported into a frame
                # Read in the initial database 2011-16.  Timenow 2014
                FCPO = pd.read_csv('FCPO_Aggregated_PerDay_Data.csv')
                FCPO['Date'] = pd.to_datetime(FCPO['Date'])
                FCPO['Date'] = pd.to_datetime(FCPO['Date'], format = "%d/%m/%Y")
                #FCPO['Date'] = pd.to_datetime(FCPO['Date'], format = "%d/%m/%Y")
                FCPO.set_index('Date', inplace=True)
                FCPO['isAvailable']=0
                FCPO['isAvailable'].iloc[i]=1
                #FCPO.head(30)
                # Partition the data to years
                for yearnow in range(2014, 2017, 1):
                    dfTrain = FCPO[str(yearnow-3)]
                    dfGA = FCPO[str(yearnow-2)]
                    dfTest = FCPO[str(yearnow-1)]
                    dfTrade = FCPO[str(yearnow)]

	#Load the Rates in a Pandas Frame. All the Data Given can be converted to a CSV And imported into a frame

	def UpdateSysDate(self):
                self.System_Date += datetime.timedelta(days=1)
                i = i+1
                
                While (self.System_Date.date() < self.FCPO['Date'].iloc[i].date()):
                    self.System_Date += datetime.timedelta(days=1)
                    
        def UpdateRateAvailability(self):
                if(self.System_Date.date() < self.FCPO['Date'].iloc[i].date()):
                    self.FCPO['isAvailable'].iloc[i]=1    

        #All the Data Given can be converted to a CSV And imported into a frame
	#Audit all the Date change and Rate Change info with timestamp into a Separate Pandas Dataframe	
	def Download(pd ,File_Path):		
        #just download both audit data frame for trading and Sys Date / Rate Change
                pd.to_csv(File_Path)
		#Calculate the total holdings, Cash balance, Value of current holdings , %Profit into a separate dataframe
		# download all three CSV to the current folder location

################################################################# 2. Helper Functions (End) ##################################################################################

################################################################# 3.GA / Fuzzy Implementations for Training and Backtesting ( Start) #########################################

        def BackTesting(self,No_Of_Days):		
                #Utilise The Set of Selected Rules and the Incoming Rates to decide buy/sell/hold
		#update the Audit Pandas Data Frame for each execution
		#value Set - 
		#No_Of_Days = -60 since The Trading is carried out with previous cycle date

	#Execute this to call GA and Fuzzy logic
	def Training(self,StartDate,EndDate,NO-OF-Days):
                ##fetch a dataframe of the above days to be calculated based on above parameters and pass it to GA which calls fuzzy and returns the best rules
                dfTrain = self.FCPO['isAvailable']
		#NO-OF-Days= 60 before Go-Live and NO-OF-Days=1 after Go-Live(Jan 1 , 2014) 
		#if(System_Date=EndDate)# run the training only on the last day of the batch
			#call this method to select the top-n best rules for Trading or carry it forward to next generation
                rules = ga.GA.EvolveGA(popSize, dfTrain) #pass the dataframe for training
                Training_Done = 1
                        #Flag-off that one round of training is completed and be prepared for back-testing
		if (Training_Done= 1)
			trading_required = 1
		return rules
		
	def Trading(self):		
                #Utilise The Set of Selected Rules and the Incoming Rates to decide buy/sell/hold
		#No_Of_Days = 0 since the trading is carried out for current date
		
################################################################# 3. GA / Fuzzy Implementations for Training and Backtesting ( End) ##########################################

################################################################# 4.Backtesting and Trading (Start) ##########################################################################		
        #this is for backtesting from 2011- 2013
	def BuildTradingSystem(self):		
        for k in range(len(weights.keys())):
			self.Training()
			self.BackTesting()
			UpdateSysDateandRates(self)
	Self.Download("TrainingReport")
		
        #this is for actual trading + Training @ end of every day from 2014- 2016
	def EarnMoney(self):		
        #just download both audit data frame for trading and Sys Date / Rate Change
		#Calculate the total holdings, Cash balance, Value of current holdings , %Profit into a separate dataframe
		# download all three CSV to the current folder location
		for k in range(len(weights.keys())):
			self.SelectRules()
			self.Trading()
			UpdateSysDateandRates(self)
			self.Training()
	Self.Download("TradingReport")

################################################################# 4.Backtesting and Trading (End) ##########################################################################		
