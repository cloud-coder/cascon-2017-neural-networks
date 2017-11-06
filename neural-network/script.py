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

#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

'''
net = network.Network([784,30,10])
net.SGD(training_data, 30, 10, 3.0, test_data = test_data)

with open('networkexample.pkl', 'wb') as f:
    dill.dump(net, f)
'''

'''
test_img = np.array(test_data[20][0])
print(test_img)
test_img = test_img.flatten().reshape((28,28))
plt.figure(2)
plt.imshow(test_img, cmap="gray")
plt.title("Test Image")
plt.show(block=False)
'''

img = cleanimg("test6.png")
#print(img.size)
#print(img.reshape(784,1))

with open('networkexample.pkl', 'rb') as f:
    net = dill.load(f)

plt.figure(1)
plt.title("This is likely the number " + str(np.argmax(net.feedforward(img.reshape((784,1))))))
plt.imshow(img, cmap="gray")
plt.show()
