# GaFuzzySystem
GA Fuzzy System for Trading Crude Palm Oil Futures

## Prerequisits:

pip install "package name" for example, pip install deap

## Approach:

1) ALL GA Logic Should go into GA Class.
Variables required for GA / or any functions will be intialized in Init
If GA requires any input, Say MACD, Fuzzy Rules - They have to be defined as parameters, and should be passed into GA Funcctions
 
2) Same Goes for Fuzzy Logic

3) Trading Class implements all the necessary fuctions for the complete trading app it includes, Training, BackTesting, Trading, Download of Report, System Date update and Rate Changes

4) SkFuzzy for Fuzzy and Deap for GA

5) All the required data is stored in Pandas Data Frame only to make the development Easier

6) One Dataframe to load the Rates provided to us. The program will update "Available" column in the dataframe to Y. Only that column should be used by GA or Fuzzy to calculate MACD. Encourage either array or Pandas Dataframe to store the rules from GA. All the changes for example - Buy / Sell / Hold Decision is logged into a DataFrame with a set of values




