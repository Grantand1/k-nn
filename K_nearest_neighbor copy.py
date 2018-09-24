#Andrew Grant
#Machine Learning
#Assignment 1

''' The purpose of this assignment is to use k-NN to predict the labels
of the test data by computing the distance of the training data using
the k-NN algorithm
'''

'''The algo. check the closet k pixels by calculating the distance
then storing the distance  in ascending order'''

#Import libraries for algo.
import pandas as pd
import numpy as np
from collections import Counter

#Create a list to store the k_distance which will later be sorted with k variable
k = 3
k_distance= []


#load data
training = pd.read_csv('MNIST_training.csv')
#X = training.iloc[:,1:]
#y = training.iloc[:,0]
#Remove the column using [''] * len(training.columns)
#print(training.shape)
#print(type(training.head().values))

training.columns = [''] * len(training.columns)
testing = pd.read_csv('MNIST_test.csv')
testing.columns = [''] * len(testing.columns)

#Split data into label and feature
training_data = np.array(training.iloc[:,1:])
training_data_label= np.array(training.iloc[:,0])
#training_data_label = training_data_label.reshape(training_data_label.shape[0],1)
testing_data = np.array(testing.iloc[:,1:])
testing_data_label= np.array(testing.iloc[:,0])
#testing_data_label = testing_data_label.reshape(testing_data_label.shape[0],1)

#print(training_data.shape)
#print(training_data_label.shape)

#print(testing_data.shape)
#print(testing_data_label.shape)

#print(testing_data[0])

globaldistLabel = {}
globalcount = -1
#Calculate the Euclidean Distance of each sample
for i in testing_data:
    globalcount = globalcount + 1
    #print('testdata',i.shape)
    #The loops are used to calcualte the distances
    #and store the  distance in in current_distance

    current_distance = []
    current_label = []
    count = -1
    for j in training_data:
        count = count + 1
        distance = np.sqrt(np.sum(np.square(i - j)))
        current_distance.append(distance)
        current_label.append(training_data_label[count])
    np.array(current_distance)
    #print(current_distance)
    distlabel = dict(zip(current_distance,current_label))
    #print(distlabel)
    #checknow = np.sort(distlabel.keys(), kind='quicksort', axis=1)
    sorteddistlabel = dict(sorted(distlabel.items()))
    #print(type(sorteddistlabel))
    #print(testing_data_label[globalcount])
    #ziplist = {}
    #ziplist[testing_data_label[globalcount]]= sorteddistlabel
    #print(ziplist)
    globaldistLabel[globalcount] = sorteddistlabel
    print(globaldistLabel)
    if globalcount ==2:
        break
    '''
    print(len(current_label))
    k_distance.append(current_distance)
    k_distance.append(training_data_label)
    np.array(k_distance)
    # After finding the distances, sort the matrix in ascending
    print(len(k_distance[0]))
    k_distance = np.sort(k_distance,kind='quicksort', axis=1)
    k_distance=k_distance[:,:k]
    count = Counter(k_distance[1,:])
    print(count)
    '''
    #break
































