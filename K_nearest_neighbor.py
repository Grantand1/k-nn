#Andrew Grant
#Machine Learning
#Assignment 1

'''The purpose of this assignment is to use k-NN to predict the labels
of the test data by computing the distance of the training data using
the k-NN algorithm'''


'''The algo. check the closet k pixels by calculating the distance
then storing the distance  in ascending order'''

'''Import libraries for algo.'''
import pandas as pd
import numpy as np
from collections import Counter
from itertools import islice

#***I got this function off stackoverflow and it is used find the kth element
def take(n, iterable):
    return dict(islice(iterable, n))

''' Global variables and data structures'''
#***Create a list to store the k_distance which will later be sorted with k variable
k = 9
k_distance= []
globalTesting ={}
globaldistLabel = {}
globalcount = -1
finalsumlist =[]

'''load data'''
training = pd.read_csv('MNIST_training.csv')
training.columns = [''] * len(training.columns)
testing = pd.read_csv('MNIST_test.csv')
testing.columns = [''] * len(testing.columns)

'''Split data into label and feature'''
#***Check the shape of the sliced data
training_data = np.array(training.iloc[:,1:])
training_data_label= np.array(training.iloc[:,0])
testing_data = np.array(testing.iloc[:,1:])
testing_data_label= np.array(testing.iloc[:,0])

#print(training_data.shape)
#print(training_data_label.shape)
#print(testing_data.shape)
#print(testing_data_label.shape)



'''Calculate the Euclidean Distance of each sample'''
for i in testing_data:
    #***The loops are used to calcualte the distances
    #***and store the  distance in in current_distance for further use
    globalcount = globalcount + 1
    finallist=[]
    current_distance = []
    current_label = []
    count = -1
    for j in training_data:
        count = count + 1
        distance = np.sqrt(np.sum(np.square(i - j)))
        current_distance.append(distance)
        current_label.append(training_data_label[count])
    np.array(current_distance)
    #print(current_distance) to check
    distlabel = dict(zip(current_distance,current_label))
    #print(distlabel) to check
    #*** Sort the dictionary in ascending order
    sorteddistlabel = dict(sorted(distlabel.items()))
    #*** the k varible is used to get the Kth item in the dict
    sorteddistlabel = take(k, sorteddistlabel.items())
    #print(type(sorteddistlabel))
    #*** At this step the code is finding the most reoccuring item in the dict
    a = sorteddistlabel.values()
    b = Counter(a)
    c = b.most_common(1)
    #*** The rest of code is used to calculte the accuracy of the algo.
    #*** This is done by comparing testlabel data to the predicted label
    #*** The correct prediction was dived by test predictions to cal. accuracy
    for row in c[0]:
        finallist.append(row)
    np.array(finallist)
    y =testing_data_label[globalcount]
    finallist.append(y)
    #print(finallist)
    first, last = 0, 2
    if finallist[first]== finallist[last]:
        finalsumlist.append(1)
        finalsum =sum(finalsumlist)
        accuracy = (finalsum/(len(testing_data_label)))*100

'''Results'''
print("The accuracy of the k-nn is", accuracy,"%", "when k is:", k)



































