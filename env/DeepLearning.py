# -*- coding: utf-8 -*-

from keras.datasets import cifar10
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D



(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(type(x_train))#Print the data type of y_train
print(type(y_train))#Print the data type of x_test
print(type(x_test))#Print the data type of y_test
print(type(y_test))

tt = x_train[0]

plt.imshow(x_train[0])

print('The label is:', y_train[1])

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

x_train = x_train / 255
x_test = x_test / 255