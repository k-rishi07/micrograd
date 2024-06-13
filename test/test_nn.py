from micrograd.engine import Value
from micrograd.nn import Neuron ,Layer ,MLP

model=MLP(3,[4,4,1])

xs=[[2.0,3.0,-1.0],
   [3.0,-1.0,0.5],
   [0.5,1.0,1.0],
   [1.0,1.0,-1.0]]

ys=[1.0,-1.0,-1.0,1.0] 
for _ in range(20):
    ypred= [model(x) for x in xs]
    #print(f"prediction {ypred}")

    #calculate loss 
    loss=sum(((yout-ygt)**2 for yout,ygt in zip(ypred,ys)),Value(0.0))
    print(f"loss {_}:{loss.data} pred:{[i.data for i in ypred]}")
    #flash gradient of each node 
    for p in model.parameters():
        p.grad=0.0
    #backward prop
    loss.backward()
    #update parameters
    for p in model.parameters():
        p.data+=(-0.000001)*p.grad

    
    #calculate y_pred
    ypred =[model(x).data for x in xs]
    print(ypred)