if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero import Parameter
import dezero.layers as L

x = Variable(np.array(1.0))
p = Parameter(np.array(2.0))

y = x * p

print(isinstance(p, Parameter))
print(isinstance(x, Parameter))
print(isinstance(y, Parameter))

layer = L

layer.p1 = Parameter(np.array(1.0))
layer.p2 = Parameter(np.array(2.0))

print(layer._params)
print('-------')

for name in layer._params:
    print(name, layer.__dict__[name])