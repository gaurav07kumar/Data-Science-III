# IC272 lab_2
#$Student Details:$
# Name: Gaurav Kumar
# Roll: B20197
# Mob No.: 8529143452

# importing libraries
import numpy as np   
import pandas as pd
import math
import matplotlib.pyplot as plt
import statsmodels.api as sm

# =============================================================================
# Question 1
# =============================================================================
print('1.------------------------------------------------------------')
data = pd.read_csv('daily_covid_cases.csv')

cases = data['new_cases']
#part a
plt.figure(figsize = (15,8))
plt.plot(list(cases),color = 'g')
plt.title('Corona cases')
plt.ylabel('New confirmed cases')
plt.xlabel('Month-Year')
labels = ['Feb-20','Apr-20','Jun-20','Aug-20','Oct-20','Dec-20','Feb-21','Apr-21','Jun-21','Aug-21','Oct-21']
ticks = [0,60,120,180,240,300,360,420,480,540,600]
plt.xticks(ticks,labels)
plt.show()

#part b
print(' (b).')
cases_lag0 = cases[1:]
cases_lag1 = cases[:-1]
AC = np.corrcoef(cases_lag0,cases_lag1)[1,0]
print(f'Pearson Corr.(Auto Correlation) Coeff = {AC}')
print('')

#part c
plt.figure(figsize = (8,8))
plt.scatter(cases_lag0,cases_lag1,s = 0.5,color='k')
plt.style.use('ggplot')
plt.xlabel('given time sequence')
plt.ylabel('One-day lagged time sequence')
plt.title('Scatter b/w given and one day lagged time sequence')
plt.show()

#part d
print('(d).')
AutoCorrelation = []
lag_value = [1,2,3,4,5,6]
for i in lag_value:
    cases_given = cases[i:]
    cases_lagi = cases[:-i]
    AC = np.corrcoef(cases_given,cases_lagi)[1,0]
    print(f'Pearson Corr.(Auto Correlation) Coeff for {i}-day lag= {AC}')
    AutoCorrelation.append(AC)
plt.figure(figsize = (15,8))
plt.plot(lag_value, AutoCorrelation, color = 'deeppink')
plt.xlabel('Lagged days')
plt.ylabel('Auto correlation coefficient')
plt.title('AC v/s days-lag')
plt.show()   

#part e
sm.graphics.tsa.plot_acf(cases,lags = 50)

# =============================================================================
# Question 2
# =============================================================================
print('2.------------------------------------------------------------')
print(' (a).')
test_size = 0.35 # 35% for testing
tst_sz = math.ceil(len(cases)*test_size)
train, test = cases[:len(cases)-tst_sz], cases[len(cases)-tst_sz:]

#part a
plt.figure(figsize = (15,8))
plt.plot(train)
plt.title('Corona cases(Train data)')
plt.ylabel('New confirmed cases')
plt.xlabel('Month-Year')
labels = ['Feb-20','Apr-20','Jun-20','Aug-20','Oct-20','Dec-20','Feb-21','Apr-21']
ticks = [0,60,120,180,240,300,360,420]
plt.xticks(ticks,labels)
plt.show()

plt.figure(figsize = (15,8))
plt.plot(test)
plt.title('Corona cases(Test data)')
plt.ylabel('New confirmed cases')
plt.xlabel('Month-Year')
labels = ['Feb-21','Apr-21','Jun-21','Aug-21','Oct-21']
ticks = [360,420,480,540,600]
plt.xticks(ticks,labels)
plt.show()

from statsmodels.tsa.ar_model import AutoReg
lag = 5 # The lag=5
model = AutoReg(train, lags= lag, old_names = False)
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model
print('Regression Coefficient are:')
print(coef)

#part b
print(' (b).')
#using these coefficients walk forward over time steps in test, one step each time
history = train[len(train)- lag:]
history = [history.iloc[i] for i in range(len(history))]
predictions = list() # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    dep_var = [history[i] for i in range(length-lag,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(lag):
        yhat += coef[d+1] * dep_var[lag-d-1] # Add other values
    predictions.append(yhat) #Append predictions to compute RMSE later
    obs = test.iloc[t]
    history.append(obs) # Append actual test value to history, to be used in next step.
    

plt.figure(figsize = (10,10))
plt.scatter(test, predictions, s = 0.7, color = 'k')
plt.title('Actual v/s Predicted cases')
plt.xlabel('Actual New Cases')
plt.ylabel('Predicted New Cases')
plt.show()


test_iloc = []
for i in test:
    test_iloc.append(i)
plt.figure(figsize = (15,8))
plt.plot(test_iloc, color='green')
plt.plot(predictions, color = 'blue')
plt.legend(['Actual','Predicted'])
plt.ylabel('New cases')
plt.xlabel('Month-Year')
labels = ['Feb-21','Apr-21','Jun-21','Aug-21','Oct-21']
ticks = [-25,35,95,155,215]
plt.xticks(ticks,labels)
plt.show()


from sklearn.metrics import mean_squared_error
RMSE = mean_squared_error(test_iloc, predictions, squared = False)
RMSE_per = RMSE*100/np.mean(test)
print(f'RMSE(%) = {round(RMSE_per,3)}%')

def Mean_abs_err_per(y,ycap):
    y, ycap = np.array(y), np.array(ycap)
    MAPE = np.mean(np.abs((y - ycap) / y)) * 100
    return MAPE
MAPE = Mean_abs_err_per(test_iloc, predictions)
print(f'MAPE(%) = {round(MAPE,3)}%')

# =============================================================================
# Question 3
# =============================================================================
print('3.------------------------------------------------------------')

RMSE_values = []
MAPE_values = []
lag_days = [1,5,10,15,25]
for i in lag_days:
    lag = i # The lag=i
    model = AutoReg(train, lags= lag, old_names = False)
    model_fit = model.fit() # fit/train the model
    coef = model_fit.params # Get the coefficients of AR model
    #using these coefficients walk forward over time steps in test, one step each time
    history = train[len(train)- lag:]
    history = [history.iloc[i] for i in range(len(history))]
    predictions = list() # List to hold the predictions, 1 step at a time
    for t in range(len(test)):
        length = len(history)
        dep_var = [history[i] for i in range(length-lag,length)]
        yhat = coef[0] # Initialize to w0
        for d in range(lag):
            yhat += coef[d+1] * dep_var[lag-d-1] # Add other values
        predictions.append(yhat) #Append predictions to compute RMSE later
        obs = test.iloc[t]
        history.append(obs) # Append actual test value to history, to be used in next step.

    RMSE = mean_squared_error(test_iloc, predictions, squared = False)
    RMSE_per = RMSE*100/np.mean(test)
    print(f'RMSE(%) for {i} day lag= {round(RMSE_per,3)}%')
    RMSE_values.append(RMSE_per)

    MAPE = Mean_abs_err_per(test_iloc, predictions)
    print(f'MAPE(%) for {i} day lag= {round(MAPE,3)}%')
    MAPE_values.append(MAPE)

plt.figure(figsize = (15,8))
plt.bar(lag_days, RMSE_values, color = 'c')
plt.title('RMSE v/s lag-days')
plt.xlabel('lag-days')
plt.ylabel('RMSE(%)')
plt.show()


plt.figure(figsize = (15,8))
plt.bar(lag_days, RMSE_values, color = 'deeppink')
plt.title('MAPE v/s lag-days')
plt.xlabel('lag-days')
plt.ylabel('MAPE(%)')
plt.show()
    
# =============================================================================
# Question 4
# =============================================================================
print('4.------------------------------------------------------------')
Threshold_value = 2/(len(train))**0.5
def check(list_):
    for item in list_:
        if np.abs(item) < Threshold_value:
            return False
    return True
p = 0
correlation = [] 
for i in range (1,len(train)-1,1):
    original = train[i:]
    lagged = train[:-i]
    AC = np.corrcoef(original,lagged)[1,0]
    correlation.append(AC)
    if check(correlation):
        p += 1
    else:
        p += 0
    
print(f'So, the heurristic value for optimal no. of lags = {p}')
model = AutoReg(train, lags= p, old_names = False)
model_fit = model.fit() # fit/train the model
coef = model_fit.params # Get the coefficients of AR model
#using these coefficients walk forward over time steps in test, one step each time
history = train[len(train)- p:]
history = [history.iloc[i] for i in range(len(history))]
predictions = [] # List to hold the predictions, 1 step at a time
for t in range(len(test)):
    length = len(history)
    dep_var = [history[i] for i in range(length-p,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(p):
        yhat += coef[d+1] * dep_var[p-d-1] # Add other values
    predictions.append(yhat) #Append predictions to compute RMSE later
    obs = test.iloc[t]
    history.append(obs) # Append actual test value to history, to be used in next step.

RMSE_new = mean_squared_error(test_iloc, predictions, squared = False)
RMSE_percent = RMSE_new*100/np.mean(test)
print(f'RMSE(%) for heuristic lag = {round(RMSE_percent,3)}%')

def Mean_abs_err_per(y,ycap):
    y, ycap = np.array(y), np.array(ycap)
    MAPE = np.mean(np.abs((y - ycap) / y)) * 100
    return MAPE
MAPE = Mean_abs_err_per(test_iloc, predictions)
print(f'MAPE(%) for heuristic lag = {round(MAPE,3)}%')

# =============================================================================
# Extra Work
# =============================================================================
history = train[len(train)- p:]
history = [history.iloc[i] for i in range(len(history))]
for t in range (100):
    length = len(history)
    dep_var = [history[i] for i in range(length-p,length)]
    yhat = coef[0] # Initialize to w0
    for d in range(lag):
        yhat += coef[d+1] * dep_var[p-d-1] # Add other values
    history.append(yhat) #Append predictions to compute RMSE later

plt.plot(history)
plt.show()
