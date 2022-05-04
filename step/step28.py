if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import matplotlib.pyplot as plt

def rosenbrok(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))
lr = 0.001
iters = 1000
s = [float(x0.data)]
t = [float(x1.data)]
for i in range(iters):
    print(x0, x1)

    y = rosenbrok(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad
    s.append(float(x0.data))
    t.append(float(x1.data))

plt.plot(s, t, 'o')
plt.show()