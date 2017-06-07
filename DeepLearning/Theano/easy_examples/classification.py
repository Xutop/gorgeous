import numpy as np
import theano
import theano.tensor as T

rng = np.random

def compute_accuracy(y_target, y_predict):
    correct_prediction = np.equal(y_target, y_predict)
    accuracy = np.sum(correct_prediction)/len(correct_prediction)
    return accuracy

N = 400
feats = 784

D = (rng.randn(N, feats),rng.randint(size=N, low=0, high=2))

x = T.dmatrix('x')
y = T.dvector('y')

w = theano.shared(rng.randn(feats), name='w')
b = theano.shared(0.1, name='b')

forward = T.nnet.sigmoid(T.dot(x, w) + b)
prediction = forward > 0.5
xent = -(y * T.log(forward) + (1-y) * T.log(1-forward))
cost = T.mean(xent) + T.sum(0.01 * (w**2))

g_w, g_b = T.grad(cost, [w, b])
lr = 0.1
train = theano.function(inputs = [x, y],
                        outputs = [prediction, cost],
                        updates = ((w, w - lr * g_w),
                                   (b, b - lr * g_b)))
predict = theano.function(inputs = [x], outputs = prediction)

for i in range(500):
    pred, err = train(D[0], D[1])
    if i%50 == 0:
        print('cost:%s' % err)
        print('accuracy:%s%%' % (compute_accuracy(D[1], predict(D[0]))*100))
