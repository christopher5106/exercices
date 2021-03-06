import cntk as C
import numpy as np

dataset_size = 200000
X = np.random.rand(dataset_size, 2)
labels = np.zeros((dataset_size, 3))
labels[X[:, 0] > X[:,1]] = [0,0,1]
labels[X[:, 0] <= X[:,1]] = [1,0,0]
labels[X[:,1] + X[:, 0] > 1] = [0, 1, 0]

init = C.initializer.normal(0.01)

theta1 = C.Parameter(shape=(2, 12), init=init )
bias1 = C.Parameter(shape=(1, 12), init=init )

theta2 = C.Parameter(shape=(12,3), init=init )
bias2 = C.Parameter(shape=(1, 3,), init=init )

x = C.input_variable(shape=(2,), needs_gradient=False)
t = C.input_variable(shape=(3,), needs_gradient=False)

def forward(x):
    y = C.times(x, theta1) + C.squeeze(bias1,0)
    y = C.element_max(y, 0.)
    return C.times(y, theta2) + C.squeeze(bias2,0)

def softmax(x):
    e = C.exp(x)
    s = C.reduce_sum(e, axis=0)
    return e/s

def crossentropy(y, t):
    prob = C.squeeze(C.reduce_sum(y*t, axis=0), 0)
    return - C.reduce_mean(C.unpack_batch(C.log(prob)))
y = crossentropy(softmax(forward(x)),t)

batch_size = 20
for i in range(min(dataset_size, 100000) // batch_size ):
    lr = 0.5 * (.1 ** ( max(i - 100 , 0) // 1000))
    sample = X[batch_size*i:batch_size*(i+1)]
    target = labels[batch_size*i:batch_size*(i+1)]
    g = y.grad({x:sample, t:target}, wrt=[theta1, bias1, theta2, bias2])
    for param,grad in g.items():
        param.value = param.value - grad * lr
    loss = y.eval({x:sample, t:target})
    print("cost {} - learning rate {}".format(loss, lr))

y = C.squeeze(C.argmax(forward(x), 0),0)
accuracy = 0
for i in range(1000):
    sample = X[batch_size*i:batch_size*(i+1)]
    target = labels[batch_size*i:batch_size*(i+1)]
    tt = y.eval({x:sample})
    accuracy += np.sum(tt == np.argmax(target, axis=1))

print("Accuracy", accuracy / 1000. /batch_size)
# accuracy 99.36
