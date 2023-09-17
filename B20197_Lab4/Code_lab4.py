# IC272 lab_2
#$Student Details:$
# Name: Gaurav Kumar
# Roll: B20197
# Mob No.: 8529143452

# importing libraries
import numpy as np   
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score




# =============================================================================
# Question 1
# =============================================================================
print('1.------------------------------------------------------------')
data = pd.read_csv('SteelPlateFaults-2class.csv')  #reading csv file into pd.dataframe

data_0 = data[data['Class'] == 0]       #drop class column
data_1 = data[data['Class'] == 1]       #drop class column
X0 = data_0.iloc[:,:-1]
X0_label = data_0.iloc[:,27]            #label
[X0_train, X0_test, X0_label_train, X0_label_test] = \
    train_test_split(X0, X0_label, test_size=0.3, random_state=42, shuffle=True)   # test train split of class 0

X1 = data_1.iloc[:,:-1]
X1_label = data_1.iloc[:,27]
[X1_train, X1_test, X1_label_train, X1_label_test] = \
    train_test_split(X1, X1_label, test_size=0.3, random_state=42, shuffle=True)   # test train split of class 1

#adding class 0 and class 1 dataset
X = pd.concat([X0,X1])
X_train = pd.concat([X0_train,X1_train])
X_test = pd.concat([X0_test,X1_test])
X_label_train = pd.concat([X0_label_train,X1_label_train])
X_label_test = pd.concat([X0_label_test,X1_label_test])

# save the train and test dataftrame
X_train.to_csv('SteelPlateFaults-train.csv',index=False)
X_test.to_csv('SteelPlateFaults-test.csv',index=False)


# using classifier to preding test data after using classifier of tarin dataset
# KNN classifier
for i in [1,3,5]:
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train,X_label_train) 
    X_test_label_pred = classifier.predict(X_test)
    accuracy = accuracy_score(X_label_test,X_test_label_pred)  # finding accuracy
    matrix = confusion_matrix(X_label_test,X_test_label_pred)  # finding confusion matrix
    print(f'Confusion Matrix for K= {i}:\n{matrix}')
    print(f'classification accuracy = {accuracy}')
    print('')
print('**The value of K for which accuracy is high: 3')
   
# =============================================================================
# Question 2 
# =============================================================================
print('2.------------------------------------------------------------')
columns_data = list(data.columns)
columns_data.remove('Class')        # removing Class
from sklearn.preprocessing import MinMaxScaler        #normalising data from sklearn inbuilt function
data_train = pd.read_csv('SteelPlateFaults-train.csv')
scaler = MinMaxScaler()
scaler.fit(data_train)
X_train_normalised = scaler.transform(data_train)
X_train_normalised = pd.DataFrame(X_train_normalised,columns= columns_data)
X_train_normalised.to_csv('SteelPlateFaults-train-Normalised.csv',index = False)

data_test = pd.read_csv('SteelPlateFaults-test.csv')
# normalising test dataset with min max of train dataset
normalised_data = {}
for i in data_test:
    normalised = []
    max_value = np.max(data_train[i])
    min_value = np.min(data_train[i])
    normalised = (data_test[i]-min_value)/(max_value-min_value) #normalizing data
    normalised_data[i] = normalised
X_test_normalised = pd.DataFrame(normalised_data)
X_test_normalised.to_csv('SteelPlateFaults-test-Normalised.csv',index = False)  # saving DataFrame as csv

# using classifier to preding test data after using classifier of tarin dataset
# KNN  normalised classifier
for i in [1,3,5]:
    classifier3 = KNeighborsClassifier(n_neighbors=i)
    classifier3.fit(X_train_normalised ,X_label_train) 
    X_test_normalised_label_pred = classifier3.predict(X_test_normalised)
    matrix = confusion_matrix(X_test_normalised_label_pred,X_label_test)
    accuracy = accuracy_score(X_label_test,X_test_normalised_label_pred)
    print(f'Confusion Matrix for K= {i}:\n{matrix}')
    print(f'classification accuracy = {accuracy}')
    print('')
print('**The value of K for which accuracy is high: 3')

# =============================================================================
# Question 3 
# =============================================================================
print('3.------------------------------------------------------------')
# X0_train is dataframe contaning train dataset off class 0.
# X1_train is dataframe contaning train dataset off class 1.

# new train dataset of class 1
X0_train.drop( ['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], axis=1, inplace=True)   
# new train dataset of class 1
X1_train.drop( ['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], axis=1, inplace=True)

# mean vector of class 1 and 0
mean_vector_0 = X0_train.mean()
mean_vector_1 = X1_train.mean()

# covariance matrices of trained data of class 0 and 1
cov_matrix_0 = X0_train.cov()
cov_matrix_1 = X1_train.cov() 

# storing covariance matrices into csv files
cov_matrix_0.to_csv('covariance matrices of class 0.csv', encoding='utf-8')
cov_matrix_1.to_csv('covariance matrices of class 1.csv', encoding='utf-8')


# new test dataset
X_test_new = X_test.drop(columns = ['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], axis=1, inplace=True) 


d = 23
N = len(X_train)
# defining prior fun to find prior of each class
def prior(class_train):
    N0 = len(X0_train)
    N1 = len(X1_train)
    if class_train is X0_train:
        prior_value = N0/N
    elif class_train is X1_train:
        prior_value = N1/N
    return prior_value
prior_0 = prior(X0_train)    
prior_1 = prior(X1_train)

# likelihood=e^(-0.5*((x-mean)T)*(cov_matrix^-1)*(x-mean))/(2pi^(d/2)*(det(cov_matrix)**0.5), d=dimension of data
# likelihood of class 0 
def likelihood(x_sample,mean_vect,cov):
    deter = np.linalg.det(cov)
    M_D_subproduct = np.dot(np.transpose(x_sample - mean_vect), np.linalg.inv(cov))
    M_D = np.dot(M_D_subproduct, (x_sample - mean_vect))
    likelihood = np.exp(-0.5 * M_D) / (((2 * np.pi) ** 11.5) * (deter ** 0.5))
    return likelihood

# Using for loop to find byers_label_pridiction
X_test_byers_label_pred = []
for k in range(len(X_test)):
    x = X_test.iloc[k]
# likelihood of each class
    like_0 = likelihood(x,mean_vector_0,cov_matrix_0)
    like_1 = likelihood(x,mean_vector_1,cov_matrix_1)  
    p_x = (like_0 * prior_1) + (like_1 * prior_1)
# prosterior probability 
    prob_0 = like_0 * prior_0/p_x
    prob_1 = like_1 * prior_1/p_x
    
    # test sample belongs to that class whose probability is high
    if prob_0 > prob_1:
        test_pred = 0
    else:
        test_pred = 1
    X_test_byers_label_pred.append(test_pred)
        
    
matrix = confusion_matrix(X_test_byers_label_pred,X_label_test)  # confusion matrix
accuracy = accuracy_score(X_label_test,X_test_byers_label_pred)  # accuracy
    
print(f'Confusion Matrix for K= {i}:\n{matrix}')
print(f'classification accuracy = {accuracy}')
print('')
print('**The value of K for which accuracy is high: 3')
    


    
    


    
    




 


