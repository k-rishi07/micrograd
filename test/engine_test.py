# manual backpropogation example single neuron f(x)=w1*x1+w2*x2+b
from micrograd.engine import Value
from graphviz import graphviz_functions as gf

    #weights 
w1=Value(-3.0,label='w1')
w2=Value(1.0,label='w2')
    #input
x1=Value(2.0,label='x1')
x2=Value(0.0,label='x2')
    #bias
b=Value(6.88,label='b')
    
w1x1=w1*x1               ; w1x1._label='w1*x1'
w2x2=w2*x2               ; w2x2._label='w2*x2' 
w1x1w2x2=w1x1+w2x2       ; w1x1w2x2._label='w1x1+w2x2' 
z=w1x1w2x2+b             ; z._label='z'
#define activation function tanh
#f=z.tanh()               ; f._label='f'# dtanhx/dx = 1-(tanhx)**2

ex=(2*z).exp()
f=(ex-1)/(ex+1) ;f._label='f'

f.backward()
gf.draw_dot(f)