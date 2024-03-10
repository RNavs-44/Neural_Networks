import torch

words = open('makemore/names.txt', 'r').read().splitlines()

# counting bigrams in python dictionary
b = {}
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

P = N.float()
# P is torch.Size([27, 27])
# P.sum(1, keepdim=True) is torch.Size([27,1])
# Broadcasting(whether 2 arrays can be combined in binary operation)
# Broadcasting semantics:
# each tensor has at least 1 dimension
# when iterating over dimension sizes, starting at trailing
# dimension sizes must either be =, one of them 1, or one doesn't exist
P /= P.sum(1, keepdim=True)
for i in range(20):
    out = []
    ix = 0
    while True:
        p = P[ix]
        # draws samples from probability distribution
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
