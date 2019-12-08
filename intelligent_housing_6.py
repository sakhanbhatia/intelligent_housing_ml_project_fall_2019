from __future__ import absolute_import, division, print_function, unicode_literals

from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
#from sklearn.tree import DecisionTreeClassifier
#from skmultilearn.adapt import MLkNN

import os
import tensorflow as tf
#from tensorflow import keras
#from keras.layers import Dense
#from keras.models import Sequential
#from keras.layers import Activation
#from keras.layers import BatchNormalization
#from keras.layers import Dropout

import numpy as np
from numpy import loadtxt

from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss

from sklearn.metrics import accuracy_score

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

file_name = 'D:\\BU Classes\\METCS767 - Machine Learning - Eric Braude\\Project\\Converted_Date_Time\\Appliance_data-full_converted_weekday_time_7days.csv'
features = loadtxt(file_name, delimiter=',',usecols = [0,1,2,3,4,5,6,7,8,9,10], skiprows = 1) #Sun, Mon, Tue, Wed, Thu, Fri, Sat, Morning, Afternoon, Evening, Night
labels = loadtxt(file_name, delimiter=',', usecols = [11,12,13,14,15,16,17,18,19,20], skiprows = 1) # television, fan, fridge, laptop/computer, electric heating, oven, washing machine, microwave, toaster, cooker


appliances = ["Television", "Fan",	"Refrigerator",	"Laptop/Computer",	"Electric Heating Element",	"Oven",	"Washing Machine",	"Microwave", "Toaster",	"Cooker"]
weekday_name =["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
time_of_day = ["Morning", "Afternoon", "Evening", "Night"]

weekday = [0,0,0,0,0,0,0]
time = [0,0,0,0]

''' #Printing the data and data shape
print(features)
print(features.shape)

print(labels)
print(labels.shape)
'''

#splitting of data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(features, labels,test_size=0.20)

'''
#Printing the training data and shape

print(X_train.shape)
print(X_test.shape)

#Printing the testing data and shape

print(y_train.shape)
print(y_test.shape)

'''



#initialize binary relevance multi-label classifier
#with a gaussian naive bayes base classifier
print("\n\nTraining data with Binary Relevance using Gaussian Naive Bayes")
classifier = BinaryRelevance(GaussianNB())

# train for Binary Relevance
classifier.fit(X_train, y_train)


# predict for Binary Relevance
predictions = classifier.predict(X_test)

#Hamming Loss for Binary Relevance
hamm_loss_binary = hamming_loss(y_test, predictions)

print("Hamming Loss:", hamm_loss_binary)



#Live Prediction
while True:
	d= int(input("\n\n\nEnter day (Sun=1, Mon=2, etc.,):"))
	h= int(input("Enter hour (24 hour format):"))

	if(d>=1 and d<=7):
		weekday[d-1] =1
	else:
		print("Invalid Day\n")
		continue

	if (h>=6 and h<=11):
		time[0] =1
		day_time = time_of_day[0]
	elif (h>=12 and h<=17):
		time[1]=1
		day_time = time_of_day[1]
	elif (h>=18 and h<=23):
		time[2]=1
		day_time = time_of_day[2]
	elif (h>=0 and h<=5):
		time[3] = 1
		day_time = time_of_day[3]
	else:
		print("Invalid time\n")
		continue
	
	#print (weekday, time)

	join_input = weekday + time
	#print(join_input)

	x = np.array(join_input)

	x.shape = (1,11)
	
	'''
	print(x.shape)
	print(x)
	'''

	prediction = classifier.predict(x)
	#print("Prediction: ", prediction)
	

	my_string = "\nPrediction for \"{}\" \"{}\" :"
	print (my_string.format(weekday_name[d-1], day_time))
	
	arr = prediction.toarray()
	
	print("The appliance(s) switched on at this time :")
	counter = 0
	for j in range(9):
		if(arr.item(j) == 1):
			print(appliances[j], end=", ")
			counter +=1

	print("\n\nTotal appliances switched on:", counter)

	
	c = int(input("Continue Prediction(1), Stop Prediction(any other input) ? :"))
	if(c!=1):
		break
	

#initialize Classifier Chains multi-label classifier
#with a gaussian naive bayes base classifier
print("\n\n\nTraining data with Classifier Chains using Gaussian Naive Bayes")
classifier = ClassifierChain(GaussianNB())

# train for Classifier Chaines
classifier.fit(X_train, y_train)

# predict for Classifier Chains
predictions = classifier.predict(X_test)


#Hamming Loss for Classifier Chaines
hamm_loss_cc = hamming_loss(y_test, predictions)

print("Hamming Loss:", hamm_loss_cc)


#Live Prediction
while True:
	d= int(input("\n\n\nEnter day (Sun=1, Mon=2, etc.,):"))
	h= int(input("Enter hour (24 hour format):"))

	if(d>=1 and d<=7):
		weekday[d-1] =1
	else:
		print("Invalid Day\n")
		continue

	if (h>=6 and h<=11):
		time[0] =1
		day_time = time_of_day[0]
	elif (h>=12 and h<=17):
		time[1]=1
		day_time = time_of_day[1]
	elif (h>=18 and h<=23):
		time[2]=1
		day_time = time_of_day[2]
	elif (h>=0 and h<=5):
		time[3] = 1
		day_time = time_of_day[3]
	else:
		print("Invalid time\n")
		continue
	
	#print (weekday, time)

	join_input = weekday + time
	#print(join_input)

	x = np.array(join_input)

	x.shape = (1,11)
	
	'''
	print(x.shape)
	print(x)
	'''

	prediction = classifier.predict(x)
	#print("Prediction: ", prediction)
	

	my_string = "\nPrediction for \"{}\" \"{}\" :"
	print (my_string.format(weekday_name[d-1], day_time))
	
	arr = prediction.toarray()
	
	print("The appliance(s) switched on at this time :")
	counter = 0
	for j in range(9):
		if(arr.item(j) == 1):
			print(appliances[j], end=", ")
			counter +=1

	print("\n\nTotal appliances switched on:", counter)

	c = int(input("Continue Prediction(1), Stop Prediction(any other input) ? :"))
	if(c!=1):
		break
	

#initialize Label Powerset multi-label classifier
#with a gaussian naive bayes base classifier
print("\n\n\nTraining data with Label Powerset using Gaussian Naive Bayes")
classifier = LabelPowerset(GaussianNB())

# train for Label Powerset
classifier.fit(X_train, y_train)


# predict for Label Powerset
predictions = classifier.predict(X_test)


#Hamming Loss for Label PowerSet
hamm_loss_lp = hamming_loss(y_test, predictions)

print("Hamming Loss:", hamm_loss_lp)


#Live Prediction
while True:
	d= int(input("\n\n\nEnter day (Sun=1, Mon=2, etc.,):"))
	h= int(input("Enter hour (24 hour format):"))

	if(d>=1 and d<=7):
		weekday[d-1] =1
	else:
		print("Invalid Day\n")
		continue

	if (h>=6 and h<=11):
		time[0] =1
		day_time = time_of_day[0]
	elif (h>=12 and h<=17):
		time[1]=1
		day_time = time_of_day[1]
	elif (h>=18 and h<=23):
		time[2]=1
		day_time = time_of_day[2]
	elif (h>=0 and h<=5):
		time[3] = 1
		day_time = time_of_day[3]
	else:
		print("Invalid time\n")
		continue
	
	#print (weekday, time)

	join_input = weekday + time
	#print(join_input)

	x = np.array(join_input)

	x.shape = (1,11)
	
	'''
	print(x.shape)
	print(x)
	'''

	prediction = classifier.predict(x)
	#print("Prediction: ", prediction)
	

	my_string = "\nPrediction for \"{}\" \"{}\" :"
	print (my_string.format(weekday_name[d-1], day_time))
	
	arr = prediction.toarray()
	
	print("The appliance(s) switched on at this time :")
	counter = 0
	for j in range(9):
		if(arr.item(j) == 1):
			print(appliances[j], end=", ")
			counter +=1

	print("\n\nTotal appliances switched on:", counter)

	c = int(input("Continue Prediction(1), Stop Prediction(any other input) ? :"))
	if(c!=1):
		break
	


