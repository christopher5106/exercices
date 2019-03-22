import cntk as C
import numpy as np

dataset_size = 200000
X = np.random.rand(dataset_size, 2)
labels = np.zeros((dataset_size, 3))
labels[X[:, 0] > X[:,1]] = [0,0,1]
labels[X[:, 0] <= X[:,1]] = [1,0,0]
labels[X[:,1] + X[:, 0] > 1] = [0, 1, 0]

x = C.input_variable(shape=(2,), needs_gradient=False)
t = C.input_variable(shape=(3,), needs_gradient=False)

init = C.initializer.normal(0.01)
with C.layers.default_options(init=init):
    z = C.layers.Sequential([
        C.layers.Dense(12, activation=C.relu),
        C.layers.Dense(3)])

y = C.cross_entropy_with_softmax(z(x),t)
acc = C.classification_error(z(x), t)

batch_size = 20
from cntk.learners import sgd, learning_parameter_schedule
lr = learning_parameter_schedule([.5 *(.1**i) for i in range(10000)], minibatch_size=batch_size, epoch_size=1000*batch_size)
learner = sgd(z.parameters, lr)
trainer = C.Trainer(z(x), (y, acc), [learner])

for i in range(min(dataset_size, 100000) // batch_size ):
    sample = X[batch_size*i:batch_size*(i+1)]
    target = labels[batch_size*i:batch_size*(i+1)]
    trainer.train_minibatch({x:sample, t:target})
    loss = trainer.previous_minibatch_loss_average
    acc = trainer.previous_minibatch_evaluation_average
    print("cost {} - acc {} - learning rate {}".format(loss, acc, learner.learning_rate()))

y = C.argmax(z(x))
accuracy = 0
for i in range(1000):
    sample = X[batch_size*i:batch_size*(i+1)]
    target = labels[batch_size*i:batch_size*(i+1)]
    tt = y.eval({x:sample})
    accuracy += np.sum(tt == np.argmax(target, axis=1))

print("Accuracy", accuracy / 1000. /batch_size)
# accuracy 99.36
