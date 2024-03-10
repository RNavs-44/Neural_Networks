from neural_net import MLP

n = MLP(3, [4, 4, 1])
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
    ]
ys = [1.0, -1.0, -1.0, 1.0] # predicted targets

for k in range(100):
    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum([(ygt - yout)**2 for ygt, yout in zip(ys, ypred)])

    # backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # update w grad descent optimisation
    for p in n.parameters():
        p.data += -0.05 * p.grad
    
    print(k, loss.data)
print(ypred)
