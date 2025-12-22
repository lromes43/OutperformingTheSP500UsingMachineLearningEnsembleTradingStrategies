
Dec_19 = ['SMCI', 'MU', 'WBD', 'ORCL', 'PWR', 'CRWD', 'CAT', 'WMB', 'PYPL', 'GEV']
Dec_22 = ['SMCI','MO','WBD','T','WMT','KO','PYPL','CVS','CPRT','IP']

#Keep - SMCI, WBD, PYPL
for i in Dec_19: 
    if i in Dec_22: 
        print(i)
    else:
        continue


#Buy - MO, T, WMT, KO, CVS, CPRT, IP

for i in Dec_22:
    if i not in Dec_19:
        print(i)
    else:
        continue


import joblib

model = joblib.load("/Users/lukeromes/Desktop/Sp500Project/RetrainedModels/FinalBoostedOneDayClassifier.job.lib")
type(model)

model.attributes()


model.feature_names
