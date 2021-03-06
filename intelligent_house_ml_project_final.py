from __future__ import absolute_import, division, print_function, unicode_literals

from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset

import os
import tensorflow as tf

import numpy as np
from numpy import loadtxt

from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss

import matplotlib.pyplot as plt 

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

file_name = 'D:\\BU Classes\\METCS767 - Machine Learning - Eric Braude\\Project\\Converted_Date_Time\\Appliance_data-full_converted_weekday_time.csv'
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

print("\n\nTraining data with Binary Relevance using Gaussian Naive Bayes")

#initialize binary relevance multi-label classifier
#with a gaussian naive bayes base classifier
classifier_binary = BinaryRelevance(GaussianNB())

# train for Binary Relevance
classifier_binary.fit(X_train, y_train)


# predict for Binary Relevance
predictions_binary = classifier_binary.predict(X_test)

#Hamming Loss for Binary Relevance
hamm_loss_binary = hamming_loss(y_test, predictions_binary)

print("Hamming Loss:", hamm_loss_binary)




print("\n\n\nTraining data with Classifier Chains using Gaussian Naive Bayes")

#initialize Classifier Chains multi-label classifier
#with a gaussian naive bayes base classifier
classifier_cc = ClassifierChain(GaussianNB())

# train for Classifier Chaines
classifier_cc.fit(X_train, y_train)

# predict for Classifier Chains
predictions_cc = classifier_cc.predict(X_test)


#Hamming Loss for Classifier Chaines
hamm_loss_cc = hamming_loss(y_test, predictions_cc)

print("Hamming Loss:", hamm_loss_cc)




print("\n\n\nTraining data with Label Powerset using Gaussian Naive Bayes")

#initialize Label Powerset multi-label classifier
#with a gaussian naive bayes base classifier
classifier_lp = LabelPowerset(GaussianNB())

# train for Label Powerset
classifier_lp.fit(X_train, y_train)


# predict for Label Powerset
predictions_lp = classifier_lp.predict(X_test)


#Hamming Loss for Label PowerSet
hamm_loss_lp = hamming_loss(y_test, predictions_lp)

print("Hamming Loss:", hamm_loss_lp)


print("\n\n\nAll hamming loss:")
print("Binary Relevance:\n", hamm_loss_binary)
print("Classifier Chains:\n", hamm_loss_cc)
print("Label Powerset:\n", hamm_loss_lp)


objects = ('BinaryRelevance', 'ClassifierChain', 'LabelPowerset')
y_pos = np.arange(len(objects))
performance = [hamm_loss_binary, hamm_loss_cc, hamm_loss_lp]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Hamming Loss')
plt.title('Gaussian Naive Classifier Usage')

plt.show()

#Live Prediction
while True:
	classifier_select= int(input("\n\n\nChoose Classifier \nBinary Relevance=1\nClassifier Chains=2\nLabel Powerset=3\nEnter:"))
	d= int(input("Enter day (Sun=1, Mon=2, etc.,):"))
	h= int(input("Enter hour (24 hour format):"))
	
	if(classifier_select != 1 and classifier_select!=2 and classifier_select!=3):
		print("Invalid Classifier selection\n")
		print("Choosen Classifier:", classifier_select)
		continue

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

	if(classifier_select == 1):
		prediction = classifier_binary.predict(x)
	elif(classifier_select == 2):
		prediction = classifier_cc.predict(x)
	elif(classifier_select == 3):
		prediction = classifier_lp.predict(x)
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
	

