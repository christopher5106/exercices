import cntk as C
import numpy as np

dataset_size = 200000
X = np.random.rand(dataset_size, 2)
labels = np.zeros((dataset_size, 3))
labels[X[:, 0] > X[:,1]] = [0,0,1]
labels[X[:, 0] <= X[:,1]] = [1,0,0]
labels[X[:,1] + X[:, 0] > 1] = [0, 1, 0]

x = C.input_variable(shape=(-1, 2,), needs_gradient=False)
t = C.input_variable(shape=(-1, 3,), needs_gradient=False)

z = C.layers.Sequential([
    C.layers.Dense(12, activation=C.relu),
    C.layers.Dense(3)])

y = C.cross_entropy_with_softmax(z(x),t)

from cntk.learners import sgd
learner = sgd(z.parameters, [(.5 * .1**i, 1000*i) for i in range(100) ])

batch_size = 20
for i in range(min(dataset_size, 100000) // batch_size ):
    sample = X[batch_size*i:batch_size*(i+1)]
    target = labels[batch_size*i:batch_size*(i+1)]
    g = y.grad({x:sample, t:target}, wrt=z.parameters)
    learner.update(g, batch_size)
    loss = y.eval({x:sample, t:target})
    print("cost {} - learning rate {}".format(loss[0], lr))

y = C.argmax(z(x))
accuracy = 0
for i in range(1000):
    sample = X[batch_size*i:batch_size*(i+1)]
    target = labels[batch_size*i:batch_size*(i+1)]
    tt = y.eval({x:sample})
    accuracy += np.sum(tt == np.argmax(target, axis=1))

print("Accuracy", accuracy / 1000. /batch_size)
# accuracy 99.36
