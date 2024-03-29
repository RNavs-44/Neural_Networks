import torch
import torch.nn.functional as F

words = open('makemore/names.txt', 'r').read().splitlines()

# counting bigrams in python dictionary
#b = {}
#for w in words[:100]:
#    chs = ['<S>'] + list(w) + ['<E>']
#    for ch1, ch2 in zip(chs, chs[1:]):
#        bigram = (ch1, ch2)
#        b[bigram] = b.get(bigram, 0) + 1
#b = sorted(b.items(), key = lambda kv: -kv[1])

# counting bigrams using PyTorch tensors
chars = sorted(list(set(''.join(words)))) # list of all characters
stoi = {s:i+1 for i,s in enumerate(chars)} # a:1, b:2, ...
stoi['.'] = 0


# returns tensor filled with scalar value 0
N = torch.zeros((27, 27), dtype=torch.int32)

# reverse stoi
itos = {i:s for s,i in stoi.items()} 

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# returns generator object which produces pseudo random numbers
g = torch.Generator().manual_seed(2147483647)

# add 1 for model smoothing, increasing count smooths model
count = 1
P = (N+count).float()
P /= P.sum(1, keepdim=True)

# draw samples 
#for i in range(20):
#    out = []
#    ix = 0
#    while True:

#        p = P[ix]
#        or 
#        xenc = F.one_hot(xs, num_classes=27).float() 
#        logits = xenc @ w
#        counts = logits.exp()
#        probs = counts / counts.sum(1, keepdims=True) 

         # draws samples from probability distribution
#        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
#        out.append(itos[ix])
#        if ix == 0:
#            break
#    print(''.join(out))
# creating loss function using maximum likelihood estimation
# closer probability is to 1 the closer log likelihood is to 0
#log_likelihood = 0.0
#n = 0
#for w in ["arnav"]:
#    chs = ['.'] + list(w) + ['.']
#    for ch1, ch2 in zip(chs, chs[1:]):
#        ix1 = stoi[ch1]
#        ix2 = stoi[ch2]
#        prob = P[ix1, ix2]
#        logprob = torch.log(prob)
#        log_likelihood += logprob
#        n += 1
#        print(f"{ch1}{ch2}: {prob:.4f} {logprob:.4f}")
#print(f"{log_likelihood=}")
#nll = -log_likelihood
#print(f"{nll=}")
# we want to minimise average negative log likelihood
#print(f"{nll/n}")

# training a neural network
# create training set of bigrams
xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

# randomly initialise 27 neurons' weights, each neuron receives 27 inputs
w = torch.randn((27, 27), generator=g, requires_grad=True)

# gradient descent
for k in range(1):
    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float() # input to network: one-hot encoding
    logits = xenc @ w # predict log counts

    # softmax activation function:
    # way of taking outputs of neural net layer and output probabilities
    counts = logits.exp() # count equivalent to N
    probs = counts / counts.sum(1, keepdims=True) # output of neural nets, probabilbities for next character

    # loss function
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(w**2).mean()
    print(loss.item())

    # backward pass
    w.grad = None # zero gradient
    loss.backward()

    # update
    w.data += -50 * w.grad


nlls = torch.zeros(5)
for i in range(5):
    # i-th bigram
    x = xs[i].item() # input character index
    y = ys[i].item() # output character index
    print('----------------')
    print(f'bigram example {i+1}: {itos[x]}{itos[y]} (indexes {x}, {y})')
    print('input to neural net:', x)
    print('output probabilities from neural net:', probs[i])
    print('label (actual next character):', y)
    p = probs[i, y]
    print('probability assigned by net to correct character:', p.item())
    logp = torch.log(p)
    print('log likelihood:', logp.item())
    nll = -logp
    print('negative log likelihood', nll.item())
    nlls[i] = nll
print('================')
print('average negative log likelihood, i.e. loss =', nlls.mean().item())