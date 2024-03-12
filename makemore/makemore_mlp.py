import torch
import torch.nn.functional as F
import random

# read in all the words
words = open("makemore/names.txt", "r").read().splitlines()

# build vocabulary of characters of and mappings to / from integers
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}


# build the dataset
block_size = 3 # context length: how many characters we take to predict next

def build_dataset(words):
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

    return x, y

# training split, dev / validation split, test split
# 80%, 10%, 10%
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

xtr, ytr = build_dataset(words[:n1])
xdev, ydev = build_dataset(words[n1:n2])
xts, yts = build_dataset(words[n2:])

g = torch.Generator().manual_seed(214783647) # for reproducibility
c = torch.randn((27, 10), generator=g)
w1 = torch.randn((30, 200))
b1 = torch.randn(200)
w2 = torch.randn((200, 27))
b2 = torch.randn(27)
parameters = [c, w1, b1, w2, b2]

# number of parameters in total
print(sum(p.nelement() for p in parameters))

for p in parameters:
    p.requires_grad = True

lre = torch.linspace(-3, 0, 1000)
lrs = 10 ** lre

lri = []
lossi = []
stepi = []

for i in range(200000):
    # minibatch construction
    ix = torch.randint(0, xtr.shape[0], (32,))

    # forward pass
    emb = c[xtr[ix]] # (32, 3, 2)
    h = torch.tanh(emb.view(-1, 30) @ w1 + b1) # (32, 100)
    logits = h @ w2 + b2 # (32, 27)
    loss = F.cross_entropy(logits, ytr[ix])
    # print(loss.item())

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    # lr = lrs[i]
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    # lri.append(lre[i])
    lossi.append(loss.log10().item())
    stepi.append(i)

print(loss.item())

# validation loss
emb = c[xdev] # (53, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ w1 + b1) # (53, 100)
logits = h @ w2 + b2 # (53, 27)
loss = F.cross_entropy(logits, ydev)
print(loss.item())

# test loss
emb = c[xts] # (53, 3, 2)
h = torch.tanh(emb.view(-1, 30) @ w1 + b1) # (53, 100)
logits = h @ w2 + b2 # (53, 27)
loss = F.cross_entropy(logits, yts)
print(loss.item())

# sample from model
g = torch.Generator().manual_seed(214783647+10)
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = c[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ w1 + b1)
        logits = h @ w2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print(''.join(itos[i] for i in out))