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
# from sklearn.decomposition import PCA

# =============================================================================
# DATA
# =============================================================================
data = pd.read_csv('pima-indians-diabetes.csv')  #reading csv file into pd.dataframe
data_1 = data.drop(['class'], axis = 1)  # 'class' column dropped

#reading attributes
pregs = data["pregs"]
plas = data["plas"]
pres = data["pres"]
skin = data["skin"]
test = data["test"]
BMI = data["BMI"]
pedi = data["pedi"]
Age = data["Age"]
Class = data["class"]

# =============================================================================
# 1.DATA TRANSFORMATION
# =============================================================================
for i in data_1:
    sample = data_1[i]                                   #pd.Series
    Q1 = np.percentile(sample, 25)                       #finding first Quartile
    Q2 = np.percentile(sample, 50)                       #finding second Quartile(mode)
    Q3 = np.percentile(sample, 75)                       #finding third Quartile
    IQR = Q3-Q1                                    
    for j in range(768):
        if sample.iloc[j] < (Q1-(1.5 * IQR)):            #lower limit conditioning
            sample.iloc[j] = Q2
        elif sample.iloc[j] > (Q3 + (1.5*IQR)):          #upper limit conditioning
            sample.iloc[j] = Q2 
#(a) part------------------------------------------------
print('1.(a)')
new_min = 5                #min value
new_max = 12               #max value
table_before = []
table_after = []
table_before.append(['Min Value','Max Value'])
table_after.append(['Min Value','Max Value'])

normalized_data = {}
for i in data_1:
    normalized = []
    max_value = np.max(data_1[i])
    min_value = np.min(data_1[i])
    normalized = (data_1[i]-min_value)*(new_max-new_min)/(max_value-min_value) + new_min  #normalizing data
    normalized_data[i] = normalized                                                       #putting value
    table_before.append([min_value,max_value])
    table_after.append([normalized.min(),normalized.max()])

norm_data = pd.DataFrame(normalized_data)        #normalized data
print('Before Min-Max Normalization:')
print(tabulate(table_before))                    #printing initial value
print('After Min-Max Normalization:')
print(tabulate(table_after))                     #printing final value


#(b) part-------------------------------------------------
print('1.(b)')
table_before = []
table_after = []
table_before.append(['Mean','SD'])
table_after.append(['Mean','SD'])

standardized_data = {}
for i in data_1:
    mean = data_1[i].mean()         #mean
    SD = data_1[i].std()            #standard deviation
    standadized = (data_1[i]-mean)/SD     #standarization
    standardized_data[i] = standadized
    table_before.append([mean,SD])
    table_after.append([standadized.mean(),standadized.std()])
stand_data = pd.DataFrame(standardized_data) 
#printing
print('Before standardization:')
print(tabulate(table_before))
print('After standardization:')
print(tabulate(table_after))

    

# =============================================================================
# 2.Synthetic data
# =============================================================================
co_var = -3
mean = np.array([0,0])
cov_matrix = np.array([[13, co_var], [co_var, 5]])
# Generating a Gaussian bivariate distribution
# with given mean and covariance matrix
x,y = np.random.multivariate_normal( mean,cov_matrix,1000).T
synthetic_data = [list(x),list(y)]
df = pd.DataFrame(synthetic_data)

#(a)-----------------------------------------------------------------
# Plotting the generated samples
fig = plt.figure(figsize = (15,8))
plt.scatter(x, y,color = 'red')
plt.xlabel('1st variable')
plt.ylabel('2st variable')
plt.title('Bi-variate Gaussian distribution')
plt.show()

#(b)-----------------------------------------------------------------
#finding eigenvalue, eigenvector
eig_val, eig_vect = np.linalg.eig(cov_matrix)
print(f'Eigen values: {eig_val[0],eig_val[1]}')
print(f'Eigen vectors: {list(eig_vect[0]),list(eig_vect[1])}')

fig = plt.figure(figsize = (15,8))
origin = np.array([[0, 0], [0, 0]])
plt.scatter(x, y,color = 'c',alpha=0.7)
#ploting eigen vectors
plt.quiver(*origin, eig_vect[:, 0], eig_vect[:, 1], color=['red', 'green'],scale = 5)
plt.xlabel('1st variable')
plt.ylabel('2st variable')
plt.title('Bi-variate Gaussian distribution')
plt.show()

#(c)------------------------------------------------------------------
for j in[0,1]:
    fig = plt.figure(figsize = (15,8))
    origin = np.array([[0, 0], [0, 0]])
    plt.scatter(x, y,color = 'c',alpha=0.7)
    #plotting eigenvectors
    plt.quiver(*origin, eig_vect[:, 0], eig_vect[:, 1], color=['red', 'green'])
    proj_x= []
    proj_y = []
    for i in range(1000):
        u = np.array([x[i],y[i]])
        v = eig_vect[j]
        # finding norm of the vector v
        v_norm = np.sqrt(sum(v**2))    
    # Apply the formula as mentioned above
    # for projecting a vector onto another vector
    # find dot product using np.dot()
        proj = (np.dot(u, v)/v_norm**2)*v
        proj_x.append(proj[0])
        proj_y.append(proj[1])
    plt.scatter(proj_x,proj_y,color = 'k',s = 5)
    plt.xlabel('1st variable')
    plt.ylabel('2st variable')
    plt.title('Bi-variate Gaussian distribution')
    plt.show()

#(d)-------------------------------------------------------
a = []
b = []
ERROR= []
for i in range(1000):
    recon_tuple = np.array([0,0])
    for j in range(2):
        u = np.array([x[i],y[i]])
        v = eig_vect[j]
        # finding norm of the vector v
        v_norm = np.sqrt(sum(v**2))    
        # Apply the formula as mentioned above
        # for projecting a vector onto another vector
        # find dot product using np.dot()
        proj_value = (np.dot(u, v)/v_norm)    #finding principle value
        a_i = proj_value*v
        #finding reconstructed data
        recon_tuple = np.add(recon_tuple,a_i)
    a.append(recon_tuple[0])
    b.append(recon_tuple[1])
    ED = 0.0
    ERROR.append(ED)
new_data = [a,b]
recon_df = pd.DataFrame(new_data)
# print dataframe.
print('Synthethic Data:')
print(df)
print('')
print('Reconstructed data:')
print(recon_df)
print('')
print('ED error:')
print(pd.DataFrame([ERROR]))

# # =============================================================================
# # 3.PCA
# # =============================================================================
# pca = PCA(2)   # PCA with 2 dimensions
# pca.fit(stand_data)
# pca_data = pca.transform(stand_data)
# pca_data = pd.DataFrame(pca_data , columns = ['1','2'])

# # variance of data from PCA
# print("Variance of the data projected along 2 dimensions:")
# print(pca.explained_variance_,"\n")

# # eigen values of the PCA matrix
# eigen_val, eigen_vec = np.linalg.eig(stand_data.cov())
# print("The Eigenvalues of the Eigenvectors are:")
# print()
# print(eigen_val)
# print()
# print(">>> We can see that the highest eigen values match with the covariance",
#       "of the PCA reduced data.\n")

# # plot the scatter plot of the data calculated from pca
# plt.scatter(pca_data["1"], pca_data["2"], alpha = 0.5)
# plt.title("Scatter plot of PCA projected data")
# plt.show()

# # plot the eigen values in descending order
# values , vecs = np.linalg.eig(stand_data.cov())
# values.sort()
# plt.bar(list(range(len(values))), values[::-1])
# plt.title("Bar graph of the eigen values in decreasing order")
# plt.show()

# # calculate reconstruction error compared to original data by using l = 1, 2,.
# # .., 8 dimensions
# cov_mat_dim8 = pca_data.cov()
# error_data = []

# for i in range(1, 9):

#     # penerate PCA data from the input data
#     pca = PCA(i)
#     pca.fit(stand_data)
#     pca_data = pca.transform(stand_data)
#     pca_data = pd.DataFrame(pca_data, columns = list(range(i)))

#     # print covariance
#     if i != 1:
#         print("Covariance matrix of dimension", i, "\n")
#         print(pca_data.cov(), "\n")

#     # generate principal values from pca data
#     back_data = pca.inverse_transform(pca_data)
#     back_data = pd.DataFrame(back_data, columns = stand_data.keys())

#     if i == 8:
#         # covariance matrix of the principle data calculated from pca
#         cov_mat_dim8 = pca_data[pca_data.keys()].cov()
#         # there should be no change in class attribute

#     tot_err = 0

#     for j in range(len(back_data)):
#         l = 0
#         for key in stand_data.keys():
#             l += (stand_data[key][j]-back_data[key][j])**2
#         l = np.sqrt(l)
#         tot_err += l
#     error_data.append(np.array([i, tot_err/len(back_data)]))

# # plot total error vs number of dimensions
# error_data = np.array(error_data)
# error_data = pd.DataFrame(error_data, columns = ["1", "2"])
# plt.plot(error_data["1"], error_data["2"], marker = ".")
# plt.title("Plot of reconstruction error with varying reduced dimensions")
# plt.show()

# # print covariance matrix calculated from matrix obtained from pca data
# print("Covariance of original data:")
# print(stand_data.cov(), "\n")

# # print covariance matrix of original data
# print("Covariance of 8 dimensional data calculated from l = 8:")
# print(cov_mat_dim8)

    







