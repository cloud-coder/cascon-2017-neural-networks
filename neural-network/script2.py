import mnist_loader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import dill

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()


molded_data = zip(*training_data)
training_inputs = [np.squeeze(x) for x in molded_data[0]]
training_outputs = [np.squeeze(x) for x in molded_data[1]]

#print(training_outputs[0])
#check molded data sizes, might need to do a reshape or a dimension shrink?

print("Begin training")
clf = MLPClassifier(hidden_layer_sizes=(30,),activation="logistic",solver="sgd")
clf.fit(training_inputs,training_outputs)
print("End training")

with open('scikitexample.pkl', 'wb') as f:
    dill.dump(clf, f)
print("Saved to file")
