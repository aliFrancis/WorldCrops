#%%
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch import nn

gpu_id = 0

bavaria = pd.read_excel("Training_bavaria.xlsx")

#%%
# bavaria_test.drop(['Unnamed: 0'], axis=1, inplace=True)
bavaria.drop(['Unnamed: 0'], axis=1, inplace=True)

#%%
#adapt crop codex train
bavaria.loc[(bavaria.NC == 601)|(bavaria.NC == 602), 'NC'] = 600  #Kartoffeln jetzt alle Code 600
bavaria.loc[(bavaria.NC == 131)|(bavaria.NC == 476), 'NC'] = 131  #Wintergerste jetzt alle Code 131
bavaria.loc[(bavaria.NC == 411)|(bavaria.NC == 171)|(bavaria.NC == 410)|(bavaria.NC == 177), 'NC'] = 400  #Mais jetzt alle Code 400
bavaria.loc[(bavaria.NC == 311)|(bavaria.NC == 489), 'NC'] = 311  #Winterraps jetzt alle Code 311
#WW = 115
#ZR = 603
bavaria.loc[~((bavaria.NC == 600)|(bavaria.NC == 131)|(bavaria.NC == 400)|(bavaria.NC == 311)|(bavaria.NC == 115)|(bavaria.NC == 603)), 'NC'] = 1  

# #assign codes to test area
# bavaria_test.NC = bavaria_test.NC.astype(str).astype(int)
# bavaria_test.loc[(bavaria_test.NC == 410), 'NC'] = 400  #Corn
# bavaria_test.loc[~((bavaria_test.NC == 600)|(bavaria_test.NC == 131)|(bavaria_test.NC == 400)|(bavaria_test.NC == 311)|(bavaria_test.NC == 115)|(bavaria_test.NC == 603)), 'NC'] = 1  #rejection class other

#%%
codex = np.unique(bavaria.NC.values).tolist()
classes = [1,2,3,4,5,6,7]
codex2class = dict(zip(codex,classes))
class2codex = dict(zip(classes,classes))
bavaria['NC'] = bavaria['NC'].map(codex2class)

# codex = np.unique(bavaria_test.NC.values).tolist()
# classes = [1,2,3,4,5,6,7]
# codex2class = dict(zip(codex,classes))
# class2codex = dict(zip(classes,classes))
# bavaria_test['NC'] = bavaria_test['NC'].map(codex2class)

#%%
entries = bavaria.shape[0]
crop_types = bavaria.NC.unique()
bands = 16
# print(bavaria.keys())
b = np.array(bavaria)

cnt = np.zeros((7))
# for n in range(entries):
#     cnt[b[n,3]-1] += 1

data = {}
for n in range(7):
    data[n] = np.zeros((bands, 300, 14))


for m in range(7):
    cnt = 0
    fcnt = 0
    for n in range(entries):
        if(b[n,3]-1==m):
            # print(m)
            if (cnt==14):
                fcnt += 1
                cnt = 0
            data[m][:,fcnt,cnt] = b[n,4:] 
            cnt += 1

for n in range(7):
    for m in range(bands):
        for k in range(300):
            # data[n][k,m,:] = data[n][k,m,:] / (np.sum(data[n][k,m,:]))
            data[n][m,k,:] = data[n][m,k,:] / np.sqrt((np.einsum('a,a->',data[n][m,k], data[n][m,k])+0.001))


# %%
inp = torch.zeros((7,bands,300,14)).double()
target = torch.zeros((7,300))
for n in range(7):
    inp[n,:,:,0:14] = torch.tensor(data[n][:,:,0:14], dtype=float)
    target[n,:] = n
inp = torch.einsum('abcd->acbd', inp)

inp_a = torch.zeros((2100,bands,14))
target_a = torch.zeros((2100,7))
k = 0
for n in range(7):
    for m in range(300):
        inp_a[k,:,:] = inp[n,m]
        target_a[k,n] = 1
        k += 1

#%%
c_type = 1
sample = 0
band = 0

#%%
from torch.utils.data import Dataset
class data_set(Dataset):

    def __init__(self, inp, tar):
        self.inp = inp
        self.tar = tar

    def __len__(self):
        return self.inp.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.inp[idx], self.tar[idx]

set = data_set(inp_a.cuda(gpu_id), target_a.cuda(gpu_id))
loader = DataLoader(set, batch_size=1, shuffle=True)

#%%
from modules import Hopfield, HopfieldPooling, HopfieldLayer

class M1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hopfield = HopfieldLayer(
            scaling=1e12,
            input_bias=False,

            input_size=14,                           # R
            hidden_size=14,                          # W_K
            pattern_size=256,                        # W_V_1
            quantity=256,                            # W_K
            output_size=28,                           # W_V_2

            # do not project layer input
            state_pattern_as_static=True,
            stored_pattern_as_static=True,
            # pattern_projection_as_static=True,

            # # do not pre-process layer input
            # normalize_stored_pattern=False,
            # normalize_stored_pattern_affine=False,
            # normalize_state_pattern=False,
            # normalize_state_pattern_affine=False,
            # normalize_pattern_projection=False,
            # normalize_pattern_projection_affine=False,

            # # do not post-process layer output
            # disable_out_projection=True
            )
        self.Linear_m = nn.Linear(in_features=28, out_features=7)
        self.Linear_b = nn.Linear(in_features=bands, out_features=1)
        self.softmax = torch.nn.Softmax(dim=1)
    
    
    def forward(self, sample):
        x = self.hopfield(sample)
        y = self.Linear_m(x)
        y = torch.einsum('abc->acb', y)
        # y = torch.einsum('abc->acb', x)
        z = self.Linear_b(y)
        return self.softmax(z)[:,:,0]

model = M1()
model.cuda(gpu_id)
# criterion = nn.MSELoss()
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-6) #, weight_decay=4e-5)


# sample = next(iter(loader))[0]
# target = next(iter(loader))[1]
# out = model(sample)
# print(out.shape)
# print(model(sample).shape)
# print(target.shape)
#%%
epochs = 500
for ep in range(epochs):
    print(ep)
    cor = 0
    for idx, data in enumerate(loader):
        
        sample = data[0]
        target = data[1]
        out = model(sample)
        if (torch.argmax(out[0]) == torch.argmax(target[0])):
            cor += 1

        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
    #     break

    if (ep%5==0):
        print('----', ep, '----')
        # print(out)
        # print(target)
        # print(torch.argmax(out[0]), torch.argmax(target[0]))
        print(cor, '/', idx)
    # break



