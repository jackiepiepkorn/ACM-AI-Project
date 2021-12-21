from keras.datasets import mnist
from keras.datasets import fashion_mnist
from tensorflow import keras
from matplotlib import pyplot
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten
import numpy

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

for i in range(9):  
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()
 
 def create_model(input_shape = (28,28,1)):
  model = keras.Sequential([
  layers.Conv2D(filters = 32, kernel_size = 3, activation = 'relu', padding = 'same', input_shape = input_shape),
  layers.MaxPool2D(pool_size = 2),
  
  layers.Conv2D(filters = 64, kernel_size = 3, activation = 'relu', padding = 'same'),
  layers.MaxPool2D(pool_size = 2),
    
  layers.Conv2D(filters = 128, kernel_size = 3, activation = 'relu', padding = 'same'),
  layers.MaxPool2D(pool_size = 2),
    
  layers.Flatten(),
  layers.Dense(units = 54, activation = 'relu'),
  layers.Dense(units = 10, activation = 'softmax')])
    
  return model
  
 def compile_model(model, optimizer = 'adam', loss='categorical_crossentropy'):
  model.compile(optimizer=optimizer,loss=loss,metrics=["accuracy"])
  
def fitting_model(model, x, y, epoch):
  model.fit(x,y,shuffle=True, epochs=epoch)
  
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

y_train = numpy.eye(10)[y_train]
y_test = numpy.eye(10)[y_test]

model = create_model((28,28,1))
compile_model(model, 'adam', 'categorical_crossentropy')

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=150,epochs=20)
model.save("tf_cnn_fashion_mnist.model")
