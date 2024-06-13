import math
import numpy as np

class Value:
    def __init__(self,data,_children=(),_op='',label=''):
        self.data=data
        self.grad=0.0
        self._prev=set(_children)
        self._op=_op
        self._label=label
        self._backward= lambda:None
    
    def __repr__(self):
        return f"Value(data={self.data } type={type(self.data)})"
    def __add__(self,other):
        other= other  if isinstance(other,Value) else Value(other)
        out=Value(self.data+other.data,(self,other),'+')
        def _backward():
            self.grad+=1.0*(out.grad)
            other.grad+=1.0*(out.grad)
        out._backward=_backward
        return out
    def __sub__(self,other):
        return self+(-other)
    
    def __mul__(self,other):
        other= other  if isinstance(other,Value) else Value(other)
        out=Value(self.data*other.data,(self,other),'*')
        def _backward():
            self.grad+=other.data*(out.grad)
            other.grad+=self.data*(out.grad)
        out._backward=_backward
        return out
    def __pow__(self,other):
        assert isinstance(other,(int,float)),"only support int /float"
        out=Value(self.data**other,(self,),f"**{other}")
        
        def _backward():
            self.grad+=other*self.data**(other-1)*out.grad
        out._backward=_backward
        return out
    def __neg__(self):
        return self * -1
    def __rmul__(self,other):
        #2*a where 2 is integer() and a is Value object
        return self*other
    def __truediv__(self,other):
        return self * other**-1
    
    def tanh(self):
        x=self.data
        out=Value((math.exp(2*x)-1)/(math.exp(2*x)+1),(self,),'tanh')
        def _backward():
            self.grad+=(1-out.data**2)
        out._backward=_backward
        return out
    #breakout tanh
    def exp(self):
        t=math.exp(self.data)
        out=Value(t,(self,),'exp') 
        
        def _backward():
            self.grad+=t*out.grad
        out._backward=_backward
        return out
   
    
    def backward(self):
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad=1.0
        for node in reversed(topo):
            node._backward()   
            
    def _dot(self,other):
        #this is my experiment
        out=Value(np.dot(self.data,other.data,(self,other),'.dot'))
        return out