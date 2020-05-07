#!/usr/bin/env python
# coding: utf-8

# # Problem2: naïve Bayes Classifiers

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import math
import statistics
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
import seaborn as sb
import operator


# ### 1- Load the data from CSV file.

# In[2]:


iris = pd.read_csv('iris.csv')


# ### 2- Shuffle and split the dataset into training and test datasets

# In[3]:


train, test = train_test_split(iris, test_size=0.3, shuffle = True)


# ### 3- Split training data by class, compute the mean vector and variance of the feature in each class.

# In[4]:


means = train.groupby('species').mean()
variances = train.groupby('species').var()


# ### 4- Calculate the prior probabilities and Gaussian probability distribution for each features and class.

# In[5]:


# Compute the likelyhood using Gaussian probability density function
const = (1/math.sqrt(2*math.pi))
factor = const*(1/(variances**0.5))
exp_denominator = 2*(variances)

def likelihood(x, features, class_name): # x is the example
    # Initialize the value by 1
    likelihood_value = 1
    
    # Compute the likelihood for each feature given the class
    for feature in features:
        likelihood_value = (likelihood_value)*factor[feature][class_name]*math.exp(-((x[feature]-means[feature][class_name])**2)/exp_denominator[feature][class_name])
    return likelihood_value

# Compute prior by counting the frquency of each class and divide it by the dataset size
prior = train.species.value_counts()/len(train)  
prior


# ###### The data is unbalanced so we'll use the maximum posterior, not only maximum likelihood

# In[6]:


# Find the posterior by computing likelihood*prior 
def posterior(x, features, class_name):
    return likelihood(x, features, class_name)*prior[class_name]


# In[7]:


# This function will predict the class of a given example based on the training set values (means and variances)
def predict_class(x,features,classes):
    posterior_values = {}
    for class_name in classes:
        posterior_values.update({class_name:posterior(x, features, class_name)})
    return max(posterior_values.items(), key=operator.itemgetter(1))[0]  #return the class_name with the maximum posterior value


# ### 5- Make prediction on the testing set.

# In[8]:


prediction = []
features = test.columns.values[:-1]
classes = test.species.unique()

for i in range(len(test)):
    predicted_class = predict_class(test.iloc[i,:-1], features, classes)
    prediction.append(predicted_class)
    
prediction = pd.Series(prediction, index = test.index)


# ### 6- Compute the accuracy, precision, and recall.

# ###### This function will compute the accuracy,precision, and recall. Given the test and prediction values.

# In[9]:


def Compute_accuracy(y_test,y_predict,classes):
    length = len(classes)
    percision = []
    recall = []
    for i in range(length):
        true_prediction = sum((y_test == y_predict) & (y_test == classes[i]))
        total_predicted = sum(y_predict == classes[i])
        percision.append(true_prediction*100/total_predicted)
        
        total_actual = sum(y_test == classes[i])
        recall.append(true_prediction*100/total_actual)
    
    accuracy = sum(y_test == y_predict)*100/len(y_test)
    return accuracy, percision, recall


# ###### This function will print accuracy, precision, and recall in a formatted way

# In[10]:


def print_accuracy(classes, accuracy, percision, recall):
    print("precision = %.2f%%" % statistics.mean(percision))
    print("recall = %.2f%%" % statistics.mean(recall))
    print("accuracy = %.2f%%" %accuracy)


# ###### Compute the accuracy, precision, and recall of the prediction resulted from the naive bayes classifiers, and print it.

# In[11]:


y_test = test[test.columns[-1]] #The target values in the testing dataset
accuracy, percision, recall = Compute_accuracy(y_test, prediction, classes)
print_accuracy(classes, accuracy, percision, recall)


# In[12]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print("precision = %.4f%%" % precision_score(test[test.columns[-1]], prediction, average='macro'))
print("recall = %.4f%%" % recall_score(test[test.columns[-1]], prediction, average='macro'))
print("accuracy = %.4f%%" % accuracy_score(test[test.columns[-1]], prediction))


# ### 7- Plot a scatter plot of the data, coloring each data point by its class.

# In[13]:


features = iris.columns[:-1].tolist()
ax = sb.pairplot(iris ,x_vars=features, y_vars=features, palette="husl", hue = 'species', kind='scatter');


# # Problem 3: KNN Classifier

# In[14]:


import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
import statistics


# ### 1- Load the dataset from CSV file.

# In[15]:


iris = pd.read_csv('iris.csv')


# ###### Specify the target feature

# In[16]:


features = iris[iris.columns[:-1].tolist()]
target = iris["species"]


# ### 2- Divide the dataset into training and testing.
# ###### Training phase in KNN: includes storing the training dataset only

# In[17]:


X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3)


# ### 3- Train the KNN algorithm.
# ###### Define the functions that help in training the KNN algorithm

# 1- Compute similarity, for a given xq and every xi on a specific training set

# In[18]:


# Since all features values are continous, we use ecluidian distance
# The less is the difference, the more is the similarity
def compute_similarity(training_set, xq):
    square_diff = (training_set - xq)**2
    return square_diff.sum(axis=1)


# 2- Get the indexes of the k nearst neighbours, which can be found by taking the k minimum distances  

# In[19]:


def Get_neighbours(k, distance):
    distance = pd.DataFrame(distance, columns = ['distance'])
    return distance.nsmallest(k,'distance').index


# 3- Find the most dominant class in the k nearest neighoubr

# In[20]:


def Compute_argmax(y_training_set, index_list):
    return y_training_set[index_list].mode()[0]


# ### 4- Test the algorithm on the test dataset with different values of k=1,3,5,10.
# ### 5- Report the accuracy, precision, and recall.

# ###### Testing phase: Apply KNN on all the example in the testing set, store the predictions on y_predict, The training phase in KNN includes only storing the data

# In[21]:


def KNN(k,X_train, X_test, y_train, y_test):
    length = X_test.shape[0]
    y_predict = []
    for i in range(length):
        distance = compute_similarity(X_train, X_test.iloc[i,:])
        index_list = Get_neighbours(k, distance)
        predict_class = Compute_argmax(y_train, index_list)
        y_predict.append(predict_class)
    return pd.Series(y_predict, index = y_test.index)


# ###### This function will compute the accuracy,precision, and recall. Given the test and prediction values.

# In[22]:


def Compute_accuracy(y_test,y_predict,classes):
    length = len(classes)
    percision = []
    recall = []
    for i in range(length):
        true_prediction = sum((y_test == y_predict) & (y_test == classes[i]))
        total_predicted = sum(y_predict == classes[i])
        percision.append(true_prediction*100/total_predicted)
        
        total_actual = sum(y_test == classes[i])
        recall.append(true_prediction*100/total_actual)
    
    accuracy = sum(y_test == y_predict)*100/len(y_test)
    return accuracy, percision, recall


# ###### This function will print accuracy, precision, and recall in a formatted way

# In[23]:


def print_accuracy(classes, accuracy, percision, recall):
    print("precision = %.2f%%" % statistics.mean(percision))
    print("recall = %.2f%%" % statistics.mean(recall))
    print("accuracy = %.2f%%" %accuracy)


# ##### Test the algorithm with k=1,3,5,10 and report the accuracy, percision, and recall

# ###### Case1: k=1

# In[24]:


y_predict_k1 = KNN(1,X_train, X_test, y_train, y_test)
classes = y_train.unique()
accuracy, percision, recall = Compute_accuracy(y_test,y_predict_k1,classes)
print_accuracy(classes,accuracy, percision, recall)


# ###### Case2: k=3

# In[25]:


y_predict_k3 = KNN(3,X_train, X_test, y_train, y_test)
accuracy, percision, recall = Compute_accuracy(y_test,y_predict_k3,classes)
print_accuracy(classes,accuracy, percision, recall)


# ###### Case3: k=5

# In[26]:


y_predict_k5 = KNN(5,X_train, X_test, y_train, y_test)
accuracy, percision, recall = Compute_accuracy(y_test,y_predict_k5,classes)
print_accuracy(classes,accuracy, percision, recall)


# ###### Case4: k=10

# In[27]:


y_predict_k10 = KNN(10,X_train, X_test, y_train, y_test)
accuracy, percision, recall = Compute_accuracy(y_test,y_predict_k10,classes)
print_accuracy(classes,accuracy, percision, recall)


# ### 6- Report k that gives the highest accuracy.
# From the values above, we found that k=1 and k=10 gives the highest accuracy.

# ## Below is KNN computed via sklearn libraries, to compare the results

# ###### Case1: k=1

# In[28]:


from sklearn.neighbors import KNeighborsClassifier
neighbor = KNeighborsClassifier(n_neighbors = 1)
neighbor.fit(X_train,y_train)

predicted = neighbor.predict(X_test)
print("number of incorrect predictions = ", sum(~(predicted == y_predict_k1))) #detect if there's a wrong prediction
print("precision = %.4f" % precision_score(y_test, predicted, average='macro'))
print("recall = %.4f" % recall_score(y_test, predicted, average='macro'))
print("accuracy = %.4f" % accuracy_score(y_test, predicted))


# ###### Case2: k=3

# In[29]:


neighbor = KNeighborsClassifier(n_neighbors = 3)
neighbor.fit(X_train,y_train)

predicted = neighbor.predict(X_test)
print("number of incorrect predictions = ", sum(~(predicted == y_predict_k3))) #detect if there's a wrong prediction
print("precision = %.4f" % precision_score(y_test, predicted, average='macro'))
print("recall = %.4f" % recall_score(y_test, predicted, average='macro'))
print("accuracy = %.4f" % accuracy_score(y_test, predicted))


# ###### Case3: k=5

# In[30]:


neighbor = KNeighborsClassifier(n_neighbors = 5)
neighbor.fit(X_train,y_train)

predicted = neighbor.predict(X_test)
print("number of incorrect predictions = ", sum(~(predicted == y_predict_k5))) #detect if there's a wrong prediction
print("precision = %.4f" % precision_score(y_test, predicted, average='macro'))
print("recall = %.4f" % recall_score(y_test, predicted, average='macro'))
print("accuracy = %.4f" % accuracy_score(y_test, predicted))


# ###### Case4: k=10

# In[31]:


neighbor = KNeighborsClassifier(n_neighbors = 10)
neighbor.fit(X_train,y_train)

predicted = neighbor.predict(X_test)
print("number of incorrect predictions = ", sum(~(predicted == y_predict_k10))) #detect if there's a wrong prediction
print("precision = %.4f" % precision_score(y_test, predicted, average='macro'))
print("recall = %.4f" % recall_score(y_test, predicted, average='macro'))
print("accuracy = %.4f" % accuracy_score(y_test, predicted))


# ###### The results of both ways are matched
