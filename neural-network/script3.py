import mnist_loader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import dill
import matplotlib.image as mpimg
from PIL import Image
import PIL.ImageOps

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def cleanimg(filename):
    img = Image.open(filename)
    img.thumbnail((28, 28), Image.ANTIALIAS)
    npimg = np.array(img)
    print(img.size)
    grayimg = (255 - rgb2gray(npimg))/255
    grayimg = grayimg - np.percentile(grayimg,80)
    grayimg = grayimg/np.amax(grayimg)
    grayimg = np.clip(grayimg,0,1)
    return grayimg

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

with open('scikitexample.pkl', 'rb') as f:
    clf = dill.load(f)

molded_data = zip(*test_data)
test_inputs = [np.squeeze(x) for x in molded_data[0]]
test_outputs = [np.squeeze(x) for x in molded_data[1]]

MLPoutput = clf.predict(test_inputs)
predict_outputs = [np.argmax(x) for x in MLPoutput]

diff = [a - b for a, b in zip(test_outputs, predict_outputs)]
print(diff)

sum = 0

for i in diff:
    #print
    if i == 0:
        sum += 1

print(100*float(sum)/len(diff))

#print(test_inputs[0])
#print(np.squeeze(img.reshape((784,1))))

img = cleanimg("test6.png")
plt.figure(1)
plt.title("This is likely the number " + str(np.argmax(clf.predict(np.squeeze(img.reshape((784,1))).reshape(1,-1)))))
plt.imshow(img, cmap="gray")
plt.show()
#print(test_outputs)
