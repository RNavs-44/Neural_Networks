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
stoi = {s:i for i,s in enumerate(chars)} # a:0, b:1, ...
stoi['<S>'] = 26
stoi['<E>'] = 27

# returns tensor filled with scalar value 0
N = torch.zeros((28, 28), dtype=torch.int32)

# reverse stoi
itos = {i:s for s,i in stoi.items()} 

for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
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
#        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
#        plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
#    plt.axis('off');