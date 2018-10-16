import numpy as np #matrix math
import ga

#Fuzzy Logic Code Goes in here
class Fuzzy:
	def __init__(self):
        # a place to store the layers
		self.layers = [] 
        


	def Fuzzy(self, MACD,RSI):
        #Round one fuzzy to identify buy signal based on RSI, MACD
		

	def Fuzzy2(self, Trade_Signal,CashBalance):        
        #use cash balance and buy signal to deceide the volume o for trade
        return max_i[0]

#Implementation of Trading System - Updating Stocks,Cash Balance
class Trading:
	def __init__(self):
		popSize = 10
        #initialise other variables SystemDate, Trading Reqd
		Init_MarketRates()
		#load our GA and Fuzzy Models
		Training_Done= False
		Trading_Required=False
		rules = ga.GA.selectPop(popSize)
		self.Fuzzy = Fuzzy()
    
    #Load the Rates in a Pandas Frame. All the Data Given can be converted to a CSV And imported into a frame
	def Init_MarketRates(self):		
        #All the Data Given can be converted to a CSV And imported into a frame
		
    
	#call this method to select the top-n best rules for Trading or carry it forward to next generation
	def SelectRules(self):
        #Pass the required data as parameters
		rules = self.GA.Whatever(whatever).Rules
		return rules
		
	#Execute this to call GA and Fuzzy logic
	def Training(self,StartDate,EndDate,NO-OF-Days):
		#NO-OF-Days= 60 before Go-Live and NO-OF-Days=1 after Go-Live(Jan 1 , 2014) 
        #While 
		if(System_Date=EndDate)
			X = self.GA.Whatever(whatever)
			Y = self.Fuzzy.Whatever(whatever)
			Z = self.Fuzzy.Whatever2(whatever)
        #make the prediction
		if (Training_Done= True)
			trading_required = y
			rules = self.SelectRules()
        #return the predicted label
		return rules
		
	#Load the Rates in a Pandas Frame. All the Data Given can be converted to a CSV And imported into a frame
	def UpdateSysDateandRates(self):		
        #All the Data Given can be converted to a CSV And imported into a frame
		#Audit all the Date change and Rate Change info with timestamp into a Separate Pandas Dataframe
		
    def BackTesting(self,No_Of_Days):		
        #Utilise The Set of Selected Rules and the Incoming Rates to decide buy/sell/hold
		#update the Audit Pandas Data Frame for each execution
		#value Set - 
		#No_Of_Days = -60 since The Trading is carried out with previous cycle date

	def Trading(self):		
        #Utilise The Set of Selected Rules and the Incoming Rates to decide buy/sell/hold
		#No_Of_Days = 0 since the trading is carried out for current date
		
	def Download(self):		
        #just download both audit data frame for trading and Sys Date / Rate Change
		#Calculate the total holdings, Cash balance, Value of current holdings , %Profit into a separate dataframe
		# download all three CSV to the current folder location
		
	def BuildTradingSystem(self):		
        for k in range(len(weights.keys())):
			self.Training()
			self.SelectRules()
			self.BackTesting()
			UpdateSysDateandRates(self)
		Self.Download("TrainingReport")
		
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

