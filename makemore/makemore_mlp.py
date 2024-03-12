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

for w in words[:8]:
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

emb = c[x] # (53, 3, 2)
h = torch.tanh(emb.view(-1, 6) @ w1 + b1) # (53, 100)
logits = h @ w2 + b2 # (53, 27)
# counts = logits.exp()
# prob = counts / counts.sum(1, keepdim=True)
# loss = -prob[torch.arange(53), y].log().mean()
F.cross_entropy(logits, y)