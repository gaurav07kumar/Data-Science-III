# IC272 lab_2
#$Student Details:$
# Name: Gaurav Kumar
# Roll: B20197
# Mob No.: 8529143452

# importing libraries
import numpy as np   
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

#---DATA----------------------------------------------------------------------#
data3_miss = pd.read_csv('landslide_data3_miss.csv')       #csv file reading
data3_origi = pd.read_csv('landslide_data3_original.csv')  #csv file reading
d3_ori = data3_origi.drop(['dates','stationid'],axis = 1)  #droping 'dates' and 'stationid' attributes

#---1.Missing data------------------------------------------------------------#
#-> Defining dictionary containing key as attribute name
#   and value as no.of missing data of that particular attribute.
bar_data = {i:data3_miss[i].isna().sum() for i in data3_miss}                  #dictionary of missing values of each attributes with help of isna().sum()
attributes = list(bar_data.keys())                                             #attribute as keys  
frequency = list(bar_data.values())                                            #frequency as value pair
plt.figure(figsize=(15,8))                                                     #resizing plot
plt.style.use('bmh')                                                           #using style
plt.bar(attributes, frequency,color = 'deeppink')                              #plotting bar plot
plt.xlabel('Attributes')                                                       #x-axis labeling
plt.ylabel('No. of Missing Value')                                             #y-axis labeling
plt.title('No. of missing value of each atributes before deletion of tuples')  #plot titeling
plt.show()                                            #show plot


#---2.Target Attributes-------------------------------------------------------#
# 2(a)
x = data3_miss['stationid'].isna().sum()                                       # missing value of attribute 'stationid'
print(f'2.(a) Total no. of tuples deleted = {x}')  #printing
list1= []
for i in range (945):
    if data3_miss['stationid'].isna()[i]:                                      # conditioning if (i+1)th value of 'stationid' is not available
        list1.append(i)                                                        # listing value of such i
data3_new = data3_miss.drop(labels = list1)                                    # droping that row/tuple
#print(data3_new)

# 2(b)
# Attributes = {temperature','humidity','pressure','rain','lightavg/o0','lightmax','moisture'}
# Total no. of attributes(n) = 7
# n/3 = 7/3 = 2.333
# Delete (drop) the tuples (rows) having 3 or more missing attributes.
list2 = []
for i in range(926):
    d_1 = data3_new.iloc[i]                                                    # (i+1)th row
    if d_1.isna().sum() >=3:                                                   # conditioning if (i+1)th tuple have NaN value for >= 3 times
        list2.append(i)                                                        # listing such i
    
data3_new= data3_new.drop(labels = list2)   #droping such tuples
#print(data3_new)
print(f'2.(b) Total no. of tuples deleted = {len(list2)}')

#---3.Finding Missing Values--------------------------------------------------#
bar_data = {i:data3_new[i].isna().sum() for i in data3_new}
#plt.show()
print('3.')
Total_missing_values = 0
#using for loop for finding Nan value for all attributes after removing tuples in Question 2.
for i in bar_data.keys():
    print(f'  Number of missing values in {i} = {bar_data[i]}')
    Total_missing_values += bar_data[i]
print(f'  *Total no. of missing values in file = {Total_missing_values}')      # for printing total NaN value in remaining DataFrame

#---4.Experiments on filling missing values-----------------------------------#
# 4(a)
print('4(a)')
d3 = data3_new.drop(['dates','stationid'], axis = 1)                           # data after deleting tuples and droping attributes
values = {i : np.mean(data3_new[i]) for i in d3}                               # dict conataing key as attributes and value pair as mean of that attribute
d3_new = d3.fillna(value = values)                                             # data after filling missing values with mean of that attributes

Table_1 = []                                                                   # making a list to form list of lists
Table_1.append(['Attributes','Mean','Median','Mode','SD'])                     # appending attribues
for i in d3_new:
    stasts = [i,d3_new[i].mean(),d3_new[i].median(),d3_new[i].mode().iloc[0],d3_new[i].std()] #finding mean, median, mode, SD
    Table_1.append(stasts)                                                     # Appending stasts
print('Statistics for each attributes of missing data after replacing with mean value: ')
print(tabulate(Table_1))                                                       #printing stasts of missing data

Table_2 = []                                                                   # making a list to form list of lists
Table_2.append(['Attributes','Mean','Median','Mode','SD'])                     # appending attribues

for i in d3_ori:
    stasts = [i,d3_ori[i].mean(),d3_ori[i].median(),d3_ori[i].mode().iloc[0],d3_ori[i].std()]
    Table_2.append(stasts)  #Appending
print()
print('Statistics for each attributes of original data: ')
print(tabulate(Table_2))                                                       #printing stasts of original data

dict1 = {}
for i in d3:
    Na = d3[i].isna().sum()
    att = d3[i]                      #attribute as pd.Series
    list2 = list(att.index)          #indices of Series
    Bool = att.isnull()              
    SE = 0                           #Square error
    for j in list2 :
        if Bool.loc[j] == True:      #conditioning if jth loc indices have Nan
            SE += (d3_new[i].loc[j]-d3_ori[i].loc[j])**2    
    RMSE = (SE/Na)**0.5              #finding RMSE by SE
    dict1[i] = RMSE                  #dict having key as attribute and value pair as RMSE value of that attribute
print(dict1)   

#plotting bar plot through dictonary
attributes = list(dict1.keys())
RMSE_values = list(dict1.values())
plt.figure(figsize=(15,8))
plt.style.use('bmh') 
plt.bar(attributes, RMSE_values,color = 'c')
plt.xlabel('Attributes')
plt.ylabel('RMSE')
plt.title('RMSE v/s Attributes')
plt.show()

# 4(b)
#doing all same thing as part (b) just replacing NaN value with interpolation
print()
print('4.(b)')
d3_new = d3.interpolate()                      #data after filling missing values with linear interpolation

Table_1 = []  # making a list to form list of lists
Table_1.append(['Attributes','Mean','Median','Mode','SD']) # appending attribues
for i in d3_new:
    stasts = [i,d3_new[i].mean(),d3_new[i].median(),d3_new[i].mode().iloc[0],d3_new[i].std()]
    Table_1.append(stasts)  #Appending
print('Statistics for each attributes of missing data after replacing with interpolation: ')
print(tabulate(Table_1))

Table_2 = []  # making a list to form list of lists
Table_2.append(['Attributes','Mean','Median','Mode','SD']) # appending attribues

for i in d3_ori:
    stasts = [i,d3_ori[i].mean(),d3_ori[i].median(),d3_ori[i].mode().iloc[0],d3_ori[i].std()]
    Table_2.append(stasts)  #Appending
print()
print('Statistics for each attributes of original data: ')
print(tabulate(Table_2))      
        
dict1 = {}
for i in d3:
    Na = d3[i].isna().sum()
    att = d3[i]
    list2 = list(att.index)
    Bool = att.isnull()
    SE = 0
    for j in list2 :
        if Bool.loc[j] == True:
            SE += (d3_new[i].loc[j]-d3_ori[i].loc[j])**2
    RMSE = (SE/Na)**0.5
    dict1[i] = RMSE
print(dict1)   
attributes = list(dict1.keys())
RMSE_values = list(dict1.values())
plt.figure(figsize=(15,8))
plt.style.use('bmh') 
plt.bar(attributes, RMSE_values,color = 'c')
plt.xlabel('Attributes')
plt.ylabel('RMSE')
plt.title('RMSE v/s Attributes (linear interpolation technique)')
plt.show()



#---5.Outliner detection------------------------------------------------------#            
#5.(a)
#for y-axis labeling
y_labels = {'temperature':'degree in Celsius',
            'rain':'rainfall in mm'}

# plotting barplots
for i in ['temperature','rain']:
    data = d3_new[i]                                   #pd.Series
    Q1 = np.percentile(data, 25)                       #finding first Quartile
    Q2 = np.percentile(data, 50)                       #finding second Quartile(mode)
    Q3 = np.percentile(data, 75)                       #finding third Quartile
    IQR = Q3-Q1                                        #inter quartile range
    plt.figure(figsize=(10,8))      
    plt.boxplot(data = d3_new, x = i)
    plt.ylabel(y_labels[i])         # labeling y-axis
    plt.title('Boxplot before replacing outliners')
    plt.show()
    outliners = []                                     #listing outliners
    outliners_position = []                            #listing outliners iloc position
    for j in range(891):
        if data.iloc[j] < (Q1-(1.5 * IQR)):            #lower limit conditioning
            outliners.append(data.iloc[j])
            outliners_position.append(j)
        elif data.iloc[j] > (Q3 + (1.5*IQR)):          #upper limit conditioning
            outliners.append(data.iloc[j])
            outliners_position.append(j)
    print(f'Outliners of {i} : {len(outliners)}')           #printing outliners list
    print(IQR)
    print()
    #5.(b)
    for k in outliners_position:
        data.iloc[k] = Q2                              #replacing outliners with Q2(mode)
    #box plotting
    plt.figure(figsize=(10,8))
    plt.boxplot(data)
    plt.ylabel(y_labels[i])         # labeling y-axis
    plt.title('Boxplot after replacing outliners')
    plt.show()


    
#-------------------------------END-------------------------------------------#
    
    



            
