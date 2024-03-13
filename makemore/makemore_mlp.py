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
vocab_size = len(itos)

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

n_embd = 10 # dimensionality of character embedding vectors
n_hidden = 200 # no. of neurons in hidden layer of MLP

g = torch.Generator().manual_seed(214783647) # for reproducibility
c = torch.randn((27, n_embd), generator=g)
w1 = torch.randn((n_embd * block_size, n_hidden))
b1 = torch.randn(n_hidden)
w2 = torch.randn((n_hidden, 27))
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

max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):
    # minibatch construction
    ix = torch.randint(0, xtr.shape[0], (batch_size,))
    xb, yb = xtr[ix], ytr[ix]

    # forward pass
    emb = c[xb] # embed characters into vectors
    embcat = emb.view(emb.shape[0], -1) # concatenate vectors
    hpreact = embcat @ w1 + b1 # hidden layer pre-activation
    h = torch.tanh(hpreact) # hidden layer
    logits = h @ w2 + b2 # output layer
    loss = F.cross_entropy(logits, yb) # loss function
    # print(loss.item())

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    if i % 10000 == 0: # print every once in a while
        print(f'{i:7d} / {max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())

print(loss.item())

@torch.no_grad() # this decorator disables gradient tracking
def split_loss(split):
    x, y = {
        'train': (xtr, ytr),
        'val': (xdev, ydev),
        'test': (xts, yts)
    }[split]
    emb = c[x]
    embcat = emb.view(emb.shape[0], -1)
    hpreact = embcat @ w1 + b1
    h = torch.tan(hpreact)
    logits = h @ w2 + b2
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

split_loss('train')
split_loss('val')

# sample from model
g = torch.Generator().manual_seed(214783647+10)

for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        # forward pass neural net
        emb = c[torch.tensor([context])]
        embcat = emb.view(1, -1)
        hpreact = embcat @ w1 + b1
        h = torch.tanh(hpreact)
        logits = h @ w2 + b2
        probs = F.softmax(logits, dim=1)
        # sample from distribution
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        # shift context window and track samples
        context = context[1:] + [ix]
        out.append(ix)
        # if we sample special '.' token, break
        if ix == 0:
            break

    print(''.join(itos[i] for i in out)) # decode and print generated word