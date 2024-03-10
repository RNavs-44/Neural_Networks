import torch
import torch.nn.functional as F

words = open('makemore/names.txt', 'r').read().splitlines()

# counting trigrams in python dictionary
#t = {}
#for w in words[:100]:
#    chs = ['.'] + ['.'] + list(w) + ['.']
#    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
#        trigram = (ch1, ch2)
#        t[trigram] = t.get(trigram, 0) + 1
#t = sorted(t.items(), key = lambda kv: -kv[1])

# counting trigrams using PyTorch tensors
chars = sorted(list(set(''.join(words)))) # list of all characters
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
chars.insert(0, '.')
stoi_tri = {c1+c2:i for i,(c1,c2) in enumerate([(c1,c2) for c1 in chars for c2 in chars])}

# returns tensor filled with scalar value 0
N = torch.zeros((27**2, 27), dtype=torch.int32)
# reverse stoi
itos = {i:s for s, i in stoi.items()}
itos_tri = {i:s for s,i in stoi_tri.items()} 

for w in words:
    chs = ['.'] + ['.'] + list(w) + ['.']
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoi_tri[ch1+ch2]
        ix2 = stoi[ch3]
        N[ix1, ix2] += 1

# returns generator object which produces pseudo random numbers
g = torch.Generator().manual_seed(2147483647)

# add 1 for model smoothing, increasing count smooths model
count = 1
P = (N+count).float()
P /= P.sum(1, keepdim=True)

# draw samples 
for i in range(20):
    out = []
    ix = 0
    while True:
        p = P[ix]
#        or 
#        xenc = F.one_hot(xs, num_classes=27).float() 
#        logits = xenc @ w
#        counts = logits.exp()
#        probs = counts / counts.sum(1, keepdims=True) 
        # draws samples from probability distribution
        iy = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:
            ix = stoi_tri['.' + itos[iy]]
        else:
            ix = stoi_tri[itos_tri[ix][1] + itos[iy]]
        out.append(itos[iy])
        if iy == 0:
            break
    print(''.join(out))

# creating loss function using maximum likelihood estimation
# closer probability is to 1 the closer log likelihood is to 0
log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + ['.'] + list(w) + ['.']
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        ix1 = stoi_tri[ch1 + ch2]
        ix2 = stoi[ch3]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        #print(f"{ch1}{ch2}{ch3}: {prob:.4f} {logprob:.4f}")
print(f"{log_likelihood=}")
nll = -log_likelihood
print(f"{nll=}")
# we want to minimise average negative log likelihood
print(f"{nll/n}")

# training a neural network
# create training set of bigrams
#xs, ys = [], []

# for w in words:
#     chs = ['.'] + list(w) + ['.']
#     for ch1, ch2 in zip(chs, chs[1:]):
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         xs.append(ix1)
#         ys.append(ix2)
# xs = torch.tensor(xs)
# ys = torch.tensor(ys)
# num = xs.nelement()
# print('number of examples: ', num)

# # randomly initialise 27 neurons' weights, each neuron receives 27 inputs
# w = torch.randn((27, 27), generator=g, requires_grad=True)

# # gradient descent
# for k in range(1):
#     # forward pass
#     xenc = F.one_hot(xs, num_classes=27).float() # input to network: one-hot encoding
#     logits = xenc @ w # predict log counts

#     # softmax activation function:
#     # way of taking outputs of neural net layer and output probabilities
#     counts = logits.exp() # count equivalent to N
#     probs = counts / counts.sum(1, keepdims=True) # output of neural nets, probabilbities for next character

#     # loss function
#     loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(w**2).mean()
#     print(loss.item())

#     # backward pass
#     w.grad = None # zero gradient
#     loss.backward()

#     # update
#     w.data += -50 * w.grad
