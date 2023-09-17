# IC272 Assignment_1
# &Student Details&
#Name: Gaurav Kumar
#Roll: B20197
#Mob No.: 8529143452

# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

#---DATA----------------------------------------------------------------------#

data = pd.read_csv('pima-indians-diabetes.csv')  #reading csv file into pd.dataframe
d_1 = data.drop(['class'], axis = 1)  # 'class' column dropped

#reading attributes value as nd array
pregs = np.array(data["pregs"])
plas = np.array(data["plas"])
pres = np.array(data["pres"])
skin = np.array(data["skin"])
test = np.array(data["test"])
BMI = np.array(data["BMI"])
pedi = np.array(data["pedi"])
Age = np.array(data["Age"])
Class = np.array(data["class"])

#---1.CENTRAL TENDENCIES------------------------------------------------------#

Table_1 = []  # making a list to form list of lists
Table_1.append(['Attributes','Mean','Median','Mode','Minimum','Maximum','SD']) # appending attribues

# generating a for loop to append all data into list'Table_1'
for i in d_1:
    # Finding stasts of attributes 
    stasts = [i,d_1[i].mean(),d_1[i].median(),d_1[i].mode().iloc[0],d_1[i].min(),d_1[i].max(),d_1[i].std()]
    Table_1.append(stasts)  #Appending
    
print('Measures of central tendencies :')
print(tabulate(Table_1))  #printing list of list in a tabular form.
print()
print()

#---2.SCATTER PLOT------------------------------------------------------------#
# scatter plot b/w Age and other attributes:
for i in d_1:
    if i != 'Age': # To exclude Age v/s Age plot
        plt.figure(figsize=(15,8))
        plt.scatter(x = 'Age',y = i,color = 'c',data = d_1,s = 15)
        plt.xlabel('Age')
        plt.ylabel(i)
        plt.show()
        
# scatter plots b/w BMI and other attributes:
for i in d_1:
    if i != 'BMI': # To exclude BMI v/s BMI plot
        plt.figure(figsize=(15,8))
        plt.scatter(x = 'BMI',y = i,color = 'deeppink',data = d_1, s= 15)
        plt.xlabel('BMI')
        plt.ylabel(i)
        plt.show()

#---3.CORRELATION COEFFICIENTS------------------------------------------------#
Table_2 = [] # creating a list
Table_2.append(['Correlation','Age','BMI']) # appendind list as 1st row

for i in d_1:
    corr = [i,d_1['Age'].corr(d_1[i]),d_1['BMI'].corr(d_1[i])]  #finding remaining rows as correlations
    Table_2.append(corr) # Appending remaining rows
    
print('Correlation Coefficients :')
print(tabulate(Table_2)) #printing list of list in tabular form.


#---4.HISTOGRAMS--------------------------------------------------------------+
# histogram for 'pregs' attribute:-
# Creating histogram
bins = np.arange(0,19,1)   #defining bins
fig, ax = plt.subplots(figsize =(15, 8))  #setting plot-size

ax.hist(pregs, bins = bins,color = 'c',edgecolor= 'white') #defining histogram
plt.style.use('ggplot')              #using plot style
tick_pos = np.arange(0,18,1) + 0.5   # tick positions ; centered
tick_lab = np.arange(0,18,1)         # tick labels ; whole numbers
plt.xticks(ticks = tick_pos,labels = tick_lab)   #setting x-ticks
plt.xlabel('no. of times pregnency') # x-labelling
plt.ylabel('Frequency')              # y-labelling
plt.show()    # Show plot


# histogram for 'Skin' attribute
# Creating histogram
bins = np.arange(0,101,5)  # defining bins
fig, ax = plt.subplots(figsize =(15, 8))   #setting plot-size
ax.hist(skin, bins = bins,color = 'c',edgecolor = "white" ) #defining histogram
ax.set_xticks(bins)      #setting x-ticks
plt.style.use('ggplot')  #using plot style
plt.xlabel('triceps skin fold thickness(mm)') # x-labelling
plt.ylabel('Frequency')                       # y-labelling
plt.show()   # Show plot


#---5.CLASS HISTOGRAMS--------------------------------------------------------+
d_2 = data.groupby('class')  # grouping by class
dg_0 = d_2.get_group(0)      # class 0
dg_1 = d_2.get_group(1)      # class 1

# class 0 histogram:-
# Creating histogram
bins = np.arange(0,19,1)    #defing bins
fig, ax = plt.subplots(figsize =(15, 8))
tick_pos = np.arange(0,18,1) + 0.5   # tick positions ; centered
tick_lab = np.arange(0,18,1)         # tick labels ; whole numbers
plt.xticks(ticks = tick_pos,labels = tick_lab)   #setting x-ticks
ax.hist(data = dg_0,x = 'pregs', bins = bins,color = 'c',edgecolor= 'white')
plt.style.use('ggplot')              #using plot style
plt.xlabel('no. of times pregnency') # x-labelling
plt.ylabel('Frequency')              # y-labelling
plt.show()    # Show plot

# class 1 histogram:-
# Creating histogram
bins = np.arange(0,19,1)             #defing bins
fig, ax = plt.subplots(figsize =(15, 8))
tick_pos = np.arange(0,18,1) + 0.5   # tick positions ; centered
tick_lab = np.arange(0,18,1)         # tick labels ; whole numbers
plt.xticks(ticks = tick_pos,labels = tick_lab)   #setting x-ticks
ax.hist(data = dg_1,x = 'pregs', bins = bins,color = 'c',edgecolor= 'white')
plt.style.use('ggplot')              #using plot style
plt.xlabel('no. of times pregnency') # x-labelling
plt.ylabel('Frequency')              # y-labelling
plt.show()    # Show plot


#---BOXPLOTS------------------------------------------------------------------+
# dictionary of x-axis labels for the plots
y_labels = {'pregs':'Number of pregnancies',
            'plas':'Plasma glucose concentration 2 hours in an oral glucose tolerance test',
            'pres':'Diastolic blood pressure (mm Hg)',
            'skin':'Triceps skin fold thickness (mm)',
            'test':'2-Hour serum insulin (mu U/mL)',
            'BMI':'Body mass index (weight in kg/(height in m)^2)',
            'pedi':'Diabetes pedigree function',
            'Age':'Age (years)'}
# plotting barplots
for i in d_1:                           
    if i != 'class':                    # excluding 'class' column
        plt.figure(figsize=(10,8))      # defining plot size
        plt.boxplot(data = d_1,x = i)   #defining boxplot
        plt.ylabel(y_labels[i])         # marking x-axis labels
        plt.show()                      #show plot
        
#---END-----------------------------------------------------------------------+