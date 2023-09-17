"""
Author : Amit Maindola
Branch : Computer Science and Engineering
Phone : +91 7470985613
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import operator


########################################## PART A ###############################################################

# Question 1
df_train = pd.read_csv("SteelPlateFaults-train.csv")
df_test = pd.read_csv("SteelPlateFaults-test.csv")

df_train_class=df_train['Class']
df_test_class=df_test['Class']
df_train.drop(df_train.columns[df_train.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
df_test.drop(df_test.columns[df_test.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

df_train_0 = df_train[df_train['Class']==0]
df_train_1 = df_train[df_train['Class']==1]
df_train_0.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400','Class'], inplace=True)
df_train_1.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400','Class'], inplace=True)
df_test.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400','Class'], inplace=True)

best=0
for q in [2,4,8,16]: # For Q=2, 4, 8, 160
    print("For Q =", q)
    gmm_0 = GaussianMixture(n_components=q, covariance_type="full", reg_covar=1e-4)
    gmm_0.fit(df_train_0.values)
    gmm_1 = GaussianMixture(n_components=q, covariance_type="full", reg_covar=1e-4)
    gmm_1.fit(df_train_1.values)
    a=gmm_0.score_samples(df_test.values)
    b=gmm_1.score_samples(df_test.values)
    prediction = []
    for i in range (len(a)):
        if a[i] > b[i]:
            prediction.append(0)
        else :
            prediction.append(1)
    confMatrix = confusion_matrix(df_test_class.values, prediction)
    accuracy=accuracy_score(df_test_class, prediction)
    print("Confusion Matrix : \n", confMatrix)
    print("Accuracy Score is : ", accuracy)
    if accuracy>best:
        best=accuracy

# Question 2
# first three accuracies are calculated in lab4
print("Accuracy for KNN classifier: 89.614")
print("Accuracy for KNN classifier on Normalised data: 97.329")
print("Accuracy for Bayes classifier: 94.658")
print("Accuracy for Bayes classifier using GMM: ",round(best*100,3))



################################################# PART B ###########################################################

# SOME OF THE UTILITY FUNCTIONS TO BE USED LATER

def linear_reg(x_train_var,y_train_var,x_test_var):        # Utility function for linear regression
    x_train_var=pd.DataFrame(x_train_var)
    y_train_var=pd.DataFrame(y_train_var)
    x_test_var=pd.DataFrame(x_test_var)
    lr=LinearRegression()
    model=lr.fit(x_train_var,y_train_var)
    pred=model.predict(x_test_var)
    pred_list=[]
    for i in range(len(pred)):
        pred_list.append(pred[i][0])
    return(pred_list)

def non_linear_reg(x_train,y_train,x_test,p):       # Utility function for non-linear regression
    x_train=np.array(x_train)[:, np.newaxis]
    y_train=np.array(y_train)[:, np.newaxis]
    x_test=np.array(x_test)[:, np.newaxis]
    poly_features = PolynomialFeatures(degree=p)
    x_poly = poly_features.fit_transform(x_train)
    regressor = LinearRegression()
    regressor.fit(x_poly, y_train)
    pred = regressor.predict(poly_features.fit_transform(x_test))
    pred_list=[]
    for i in range(len(pred)):
        pred_list.append(pred[i][0])
    return(pred_list)

def mul_non_linear_reg(x_train,y_train,x_test,p):   # Utility function for multivariate non-linear regression
    poly_features = PolynomialFeatures(degree=p)
    x_poly = poly_features.fit_transform(x_train)
    regressor = LinearRegression()
    regressor.fit(x_poly, y_train)
    pred = regressor.predict(poly_features.fit_transform(x_test))
    return pred

def root_mean_squared_error(x,y):           # Utility function for Root mean squared error
    rmse=0
    x=np.array(x)
    y=np.array(y)
    for i in range(len(x)):
        rmse+=(x[i]-y[i])**2
    return ((rmse/len(x))**0.5)



# Question 1
print("\n\nQuestion 1\n")
df = pd.read_csv("abalone.csv")
# Partitioning attributes in x and x_label
column=[]
for i in df.columns:
    if i != 'Rings':
        column.append(i)
x=df[column]
x_label=df['Rings']

# Spliting the data
x_train, x_test, x_train_label, x_test_label = train_test_split(x, x_label, test_size=0.3, random_state=42, shuffle=True)

# Finding the best attribute ( Here best attribute means attribute for which pearson's correlation coefficient is maximum)
best_attribute=''
best_correlation = 0
for i in column:
    temp = df[i].corr(df['Rings'])
    if temp>best_correlation:
        best_correlation=temp
        best_attribute=i

# Question 1 part (a)
plt.scatter(x_train[best_attribute],x_train_label, label="Actual rings", color="b")
plt.title("For training data")
plt.xlabel(best_attribute)
plt.ylabel("Rings")
y=linear_reg(x_train[best_attribute],x_train_label,x_train[best_attribute]) # As here we have to use x_train[best_attribute] as testing data too.
plt.plot(x_train[best_attribute], y, color ='r', label='Predicted rings')
plt.legend()
plt.show()

# Question 1 part (b)
print("RMSE on training data: ", root_mean_squared_error(x_train_label,linear_reg(x_train[best_attribute],x_train_label,x_train[best_attribute]))) # As here we have to use x_train[best_attribute] as test data too.

# Question 1 part (c)
print("RMSE on testing data: ", root_mean_squared_error(x_test_label,linear_reg(x_train[best_attribute],x_train_label,x_test[best_attribute]))) # As here we have to use x_test[best_attribute] as test data

# Question 1 part (d)
plt.scatter(x_test_label,linear_reg(x_train[best_attribute],x_train_label,x_test[best_attribute]), color="b") # As here we have to use x_test[best_attribute] as test data
plt.title("For testing data")
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.show()


# Question 2
print("\n\nQuestion 2\n")
# Question 2 part (a)
print("RMSE on training data: ",root_mean_squared_error(x_train_label,linear_reg(x_train,x_train_label,x_train)))

# Question 2 part (b)
print("RMSE on testing data: ",root_mean_squared_error(x_test_label,linear_reg(x_train,x_train_label,x_test)))

# Question 2 part (c)
plt.scatter(x_test_label,linear_reg(x_train,x_train_label,x_test), color="b")
plt.title("For testing data")
plt.xlabel('Actual Rings')
plt.ylabel('Predicted Rings')
plt.show()


# Question 3
print("\n\nQuestion 3\n")
p=[2,3,4,5]

# Question 3 part (a)
rmse_train=[]
for i in p:
    pred=non_linear_reg(x_train[best_attribute],x_train_label,x_train[best_attribute],i)
    rmse_train.append(root_mean_squared_error(np.array(x_train_label), pred))

print("RMSE on training data for p=2,3,4,5 are: ", rmse_train)
plt.bar(p,rmse_train)
plt.xlabel("Degree of the polynomial")
plt.ylabel("RMSE")
plt.title("For training data")
plt.show()

# Question 3 part (b)
rmse_test=[]
for i in p:
    pred=non_linear_reg(x_train[best_attribute],x_train_label,x_test[best_attribute],i)
    rmse_test.append(root_mean_squared_error(np.array(x_test_label), pred))

print("RMSE on testing data for p=2,3,4,5 are: ", rmse_test)
plt.bar(p,rmse_test)
plt.xlabel("Degree of the polynomial")
plt.ylabel("RMSE")
plt.title("For testing data")
plt.show()

# Question 3 part (c)
# RMSE for training data is minimum for p=5
pred_train=non_linear_reg(x_train[best_attribute],x_train_label,x_train[best_attribute],5)
plt.scatter(x_train[best_attribute],x_train_label, label="Actual rings", color="b")
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x_train[best_attribute],pred_train), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x, y_poly_pred, color='red', label="Predicted rings")
plt.title("For Training data")
plt.xlabel("Shell weight")
plt.ylabel("Rings")
plt.legend()
plt.show()

# Question 3 part (d)
# RMSE for testing data is minimum for p=4
pred_test=non_linear_reg(x_train[best_attribute],x_train_label,x_test[best_attribute],4)
plt.scatter(x_test_label, pred_test, color="b")
plt.title("For testing data")
plt.xlabel("Actual rings")
plt.ylabel("Predicted rings")
plt.show()


# Question 4
print("\n\nQuestion 4\n")
# Question 4 part (a)
rmse_train=[]
P=[2,3,4,5]
for i in P:
    pred=mul_non_linear_reg(x_train,x_train_label,x_train,i)
    rmse_train.append(root_mean_squared_error(np.array(x_train_label), pred))
print("RMSE on training data for p=2,3,4,5 are: ", rmse_train)
plt.bar(P, rmse_train)
plt.xlabel("Degree of the polynomial")
plt.ylabel("RMSE")
plt.title("For training data")
plt.show()

# Question 4 part (b)
rmse_test=[]
for i in P:
    pred=mul_non_linear_reg(x_train,x_train_label,x_test,i)
    rmse_test.append(root_mean_squared_error(np.array(x_test_label), pred))
print("Prediction accuracy on testing data for p=2,3,4,5 are: ", rmse_test)
plt.bar(P, rmse_test)
plt.xlabel("Degree of the polynomial")
plt.ylabel("RMSE")
plt.title("For testing data")
plt.show()

# Question 4 part (c)
# RMSE value for testing data is minimum for p=2:
pred_test=mul_non_linear_reg(x_train,x_train_label,x_test,2)
plt.scatter(x_test_label, pred_test, color="b")
plt.title("For testing data")
plt.xlabel("Actual rings")
plt.ylabel("Predicted rings")
plt.show()