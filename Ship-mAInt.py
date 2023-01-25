# Import necessary libraries
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# For extracting input and output data
x = Input.T
t = Output.T

# define "classifier"
clf = MLPClassifier(hidden_layer_sizes=(10,), solver='adam')

# divide data into training and test groups
x_train, x_test, t_train, t_test = train_test_split(x, t, test_size=0.3, random_state=42)

# fit classifier to test group
clf.fit(x_train, t_train)

# create prediction of test values
t_pred = clf.predict(x_test)

# calculate model accuracy
performance = accuracy_score(t_test, t_pred)

# print classifier parameters
print(clf)

# import matplotlib.pyplot as plt
# plt.plot(performance)
# plt.show()
# Note that the functions train_test_split and accuracy_score are from the scikit-learn library
# MLPClassifier is a classifier for a neural network, the function view(net) is replaced with print(clf) which will print the classifier parameters
# You can use the matplotlib library to plot the model's performance
# When testing the code, make sure to import the "Numpy" library
# To run the program, it is necessary to insert the database for the necessary variables
