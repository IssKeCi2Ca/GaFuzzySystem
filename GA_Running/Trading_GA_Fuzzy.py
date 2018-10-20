import numpy as np #matrix math
import pandas as pd
from datetime import datetime,timedelta,date,time
#import AssetFuzzy4
#import ga

#Implementation of Trading System - Updating Stocks,Cash Balance
class Trading:
	def __init__(self):
		self.Training_Done= 0
		self.Trading_Required=0
		self.System_Date = datetime(2011, 1, 1, 00, 00)
		self.popSize = 10
		self.i=0
		self.cash_bal=10000000
		self.TradingInfo = pd.DataFrame()#to hold back testing results
		self.BackTestingInfo = pd.DataFrame()#to hold back testing results
		self.train_dict = None
		self.test_dict = None
		self.FCPO = None
		self.initMarketRates()
		print("in init")

	def initMarketRates(self):
		print("In Initrates")
		self.FCPO = pd.read_csv('FCPO_Aggregated_PerDay_Data.csv')
		self.FCPO['Date'] = pd.to_datetime(self.FCPO['Date'])
		#FCPO['Date'] = pd.to_datetime(FCPO['Date'], format = "%d/%m/%Y")
		#FCPO.set_index('Date', inplace=True)
		self.FCPO['Date'] = pd.to_datetime(self.FCPO['Date'], format = "%d/%m/%Y")
		self.FCPO.set_index('Date', inplace=True)
		self.FCPO = self.FCPO.sort_index()
		self.FCPO['isAvailable']=0
		self.FCPO['isAvailable'].iloc[self.i]=1
		# Partition the data to years
		self.train = self.FCPO['2011-01-01':'2013-12-31']
		self.train_count=self.train.shape[0]
		self.test  = self.FCPO['2014-01-01':]
		self.test_count=self.test.shape[0]
		#form a dictionary pair of dataframes for training and testing separately		
		#training set  - divided into 6 equal parts aapprox. 
		self.train_dict = {n: self.train.iloc[n:n+round(self.train_count/6), :] 
				for n in range(0, len(self.train), round(self.train_count/6))}
		#testing set  - divided into parts of 5 days
		self.test_dict = {n: self.test.iloc[n:n+5, :] 
				for n in range(0, len(self.test), 5)}
		print(self.train_dict.keys())
		#train_dict.keys()
		#len(train_dict)
		#for key in train_dict.keys
	
	def UpdateSysDate(self):
                print("in update date")
                self.System_Date += timedelta(days=1)
                self.i = self.i+1
                idx_date=self.FCPO.index
                while (self.System_Date.date() < idx_date[self.i].date()):
                        self.System_Date += timedelta(days=1)
                        #increment the system date till we reach the date for which rates data is available
	def Update_Sentiment(self):
    		print("in update sentiment")
                
	def UpdateRateAvailability(self):
    		self.FCPO['isAvailable'].iloc[self.i]=1

	def BackTesting(self):
                #FOR LOOP TO INCREMENT AND TRADE
                #---result_bt = ga.GA.CheckTrade(cash_bal,self.tradingRules)
		self.UpdateSysDate()
		self.UpdateRateAvailability()
		print("In BackTesting")
                #--self.BackTestingInfo = pd.concat([BackTestingInfo,result_bt])
		#Execute this to call GA and Fuzzy logic
	
	def Training(self,dfTrain):
		#self.tradingRules = ga.GA.selectPop(popSize, dfTrain,cash_bal) #pass the dataframe for training
		print("in training")
		self.Training_Done=1
		#Flag-off that one round of training is completed and be prepared for back-testing
		if self.Training_Done==1:
			self.trading_required = 1

	def Trading(self):
		#self.result_trade = ga.GA.CheckTrade(self.cash_bal, self.rates)
		self.UpdateSysDate()
		self.UpdateRateAvailability()
		#self.TradingInfo = pd.concat([self.TradingInfo, self.result_trade])                

	def BuildTradingSystem(self):
		print("In BuildTradingSystem")		
		for key in self.train_dict.keys():
			self.Training(self.train_dict[key])
			self.BackTesting()
		self.BackTestingInfo.to_csv("TrainingReport.csv")	
        #this is for actual trading + Training @ end of every day from 2014- 2016

	def EarnMoney(self):
		print("In EarnMoney")
		for key in self.test_dict.keys():
			for self.rates in self.test_dict[key]:
				self.Trading()
				self.UpdateSysDate()
				self.UpdateRateAvailability()
			self.Training(self.test_dict[key])
		self.TradingInfo.to_csv("TradingReport.csv")
