import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import os
import pickle

class Layer(object):
    def __init__(self, inputs, in_size, out_size, activation=None):
        self.w = theano.shared(np.random.normal(0,1,(in_size,out_size)))
        self.b = theano.shared(np.zeros(out_size,)+0.1)
        self.forward = T.dot(inputs, self.w) + self.b
        self.activation = activation
        if activation is None:
            self.outputs = self.forward
        else:
            self.outputs = self.activation(self.forward)

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) + noise

x = T.dmatrix('x')
y = T.dmatrix('y')

l1 = Layer(x, 1, 10, T.nnet.relu)
l2 = Layer(l1.outputs, 10, 1, None)

cost = T.mean(T.square(l2.outputs - y))
g_w1, g_b1, g_w2, g_b2 = T.grad(cost, [l1.w, l1.b, l2.w, l2.b])
lr = 0.05

train = theano.function([x, y], cost, 
                   updates = [(l1.w, l1.w - g_w1 * lr),
                              (l1.b, l1.b - g_b1 * lr),
                              (l2.w, l2.w - g_w2 * lr),
                              (l2.b, l2.b - g_b2 * lr)])
predict = theano.function([x], l2.outputs)

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
bx = fig.add_subplot(1,2,2)
ax.scatter(x_data, y_data)
bx.scatter(x_data, y_data)
plt.ion()
plt.show()

for i in range(1000):
    err = train(x_data, y_data)
    if i%100 == 0:
        #print(err)
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        predicted = predict(x_data)
        lines = ax.plot(x_data, predicted, 'r-', lw = 5)
        plt.pause(0.5)
        
#plt.savefig(os.getcwd()+'/linear_regression.jpg')
#
#with open('model.pkl', 'wb') as f:
#    pickle.dump(l1.w.get_value(), f)
#
#with open('model.pkl', 'rb') as f:
#    mod = pickle.load(f)
#print(mod)
