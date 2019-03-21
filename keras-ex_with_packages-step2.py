from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import RandomNormal
import numpy as np

dataset_size = 200000
X = np.random.rand(dataset_size, 2)
labels = np.zeros((dataset_size, 3))
labels[X[:, 0] > X[:,1]] = [0,0,1]
labels[X[:, 0] <= X[:,1]] = [1,0,0]
labels[X[:,1] + X[:, 0] > 1] = [0, 1, 0]

x = K.placeholder(shape=(None, 2))
t = K.placeholder(shape=(None, 3))

initializer = RandomNormal(mean=0.0, stddev=0.01, seed=None)

model = Sequential()
model.add(Dense(12, kernel_initializer=initializer, activation='relu', input_shape=(2,)))
model.add(Dense(3, kernel_initializer=initializer, activation='softmax'))

optimizer = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


model.fit(X, labels, batch_size=20, nb_epoch=1)

f = K.function([x], [K.argmax(forward(x),axis=1)])
accuracy = 0
for i in range(1000):
    sample = X[batch_size*i:batch_size*(i+1)]
    target = labels[batch_size*i:batch_size*(i+1)]
    tt = f([sample])[0]
    accuracy += np.sum(tt == np.argmax(target, axis=1))

print("Accuracy", accuracy / 1000. /batch_size)
# accuracy 99.44
