import torch
import matplotlib.pyplot as plt


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

for w in words[:1000]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

# visualising bigrams using matplotlib
#plt.figure(figsize=(16,16))
#plt.imshow(N, cmap='Blues')
#for i in range(28):
#    for j in range(28):
#        chstr = itos[i] + itos[j]
# s       plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
#        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
#    plt.axis('off');
        
# normalise counts and create probability distribution
p = N[0].float()
p = p / p.sum()

# returns generator object which produces pseudo random numbers
g = torch.Generator().manual_seed(2147483647)
# draws samples from probability distribution
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
print(itos[ix])
