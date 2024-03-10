import random
from engine import Value

class Module: # parent class to mirror nn.module class in PyTorch
    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0
    
    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, n_in, lin=False):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(random.uniform(-1, 1))
        self.lin = lin
    
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if not self.lin else act
    
    def parameters(self):
        return self.w + [self.b]
    
class Layer(Module):
    def __init__(self, n_in, n_out):
        self.neurons = [Neuron(n_in) for _ in range(n_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out)==1 else out
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module): # Multi Layer Perceptron
    def __init__(self, n_in, n_outs): # take number of inputs and list of n_outs
        sz = [n_in] + n_outs
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(n_outs))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

