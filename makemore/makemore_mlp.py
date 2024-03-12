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
    print(w)
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        x.append(context)
        y.append(ix)
        print(''.join(itos[i] for i in context), '--->', itos[ix])
        context = context[1:] + [ix] # crop and append

x = torch.tensor(x)
y = torch.tensor(y)

c = torch.randn((27, 2))
print(F.one_hot(torch.tensor(5), num_classes=27).float() @ c)
