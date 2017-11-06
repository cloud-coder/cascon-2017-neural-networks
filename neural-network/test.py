import network
import mnist_loader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import dill
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

print("loading data")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print("data loaded")

print("initializing and training network")
net = network.Network([784,30,10])
net.SGD(training_data, 10, 10, 3.0, test_data = test_data)
print("network trained")

print("saving trained network via dill")
with open('networkexample.pkl', 'wb') as f:
    dill.dump(net, f)
print("data saved")

print("loading and cleaning actual test image")
img = cleanimg("test6.png")
print("image ready")

print("loading saved network")
with open('networkexample.pkl', 'rb') as f:
    net = dill.load(f)
print("network loaded")

print("plotting image")
plt.close("all")
plt.figure(1)
plt.title("This is likely the number " + str(np.argmax(net.feedforward(img.reshape((784,1))))))
plt.imshow(img, cmap="gray")
plt.show()
