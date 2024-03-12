import torch
import torch.nn.functional as F

# read in all the words
words = open("makemore/names.txt", "r").read().splitlines()

# build vocabulary of characters of and mappings to / from integers
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

# build the dataset

block_size = 3 # context length: how many characters we take to predict next
x, y = [], []

for w in words:
    #print(w)
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        x.append(context)
        y.append(ix)
        #print(''.join(itos[i] for i in context), '--->', itos[ix])
        context = context[1:] + [ix] # crop and append

x = torch.tensor(x)
y = torch.tensor(y)

g = torch.Generator().manual_seed(214783647) # for reproducibility
c = torch.randn((27, 2), generator=g)
w1 = torch.randn((6, 100))
b1 = torch.randn(100)
w2 = torch.randn((100, 27))
b2 = torch.randn(27)
parameters = [c, w1, b1, w2, b2]

# number of parameters in total
print(sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True

for _ in range(1000):
    # minibatch construction
    ix = torch.randint(0, x.shape[0], (32,))

    # forward pass
    emb = c[x[ix]] # (53, 3, 2)
    h = torch.tanh(emb.view(-1, 6) @ w1 + b1) # (53, 100)
    logits = h @ w2 + b2 # (53, 27)
    loss = F.cross_entropy(logits, y[ix])
    # print(loss.item())

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    for p in parameters:
        p.data += -0.1 * p.grad

print(loss.item())

# evaluate loss on entire data set
emb = c[x[ix]] # (53, 3, 2)
h = torch.tanh(emb.view(-1, 6) @ w1 + b1) # (53, 100)
logits = h @ w2 + b2 # (53, 27)
loss = F.cross_entropy(logits, y[ix])
print(loss.item())