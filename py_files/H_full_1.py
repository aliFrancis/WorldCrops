#%%
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
#%%

gpu_id = 0
batch_size = 10

bavaria = pd.read_excel("Training_bavaria.xlsx")
# bavaria_test.drop(['Unnamed: 0'], axis=1, inplace=True)
bavaria.drop(['Unnamed: 0'], axis=1, inplace=True)
#adapt crop codex train
bavaria.loc[(bavaria.NC == 601)|(bavaria.NC == 602), 'NC'] = 600  #Kartoffeln jetzt alle Code 600
bavaria.loc[(bavaria.NC == 131)|(bavaria.NC == 476), 'NC'] = 131  #Wintergerste jetzt alle Code 131
bavaria.loc[(bavaria.NC == 411)|(bavaria.NC == 171)|(bavaria.NC == 410)|(bavaria.NC == 177), 'NC'] = 400  #Mais jetzt alle Code 400
bavaria.loc[(bavaria.NC == 311)|(bavaria.NC == 489), 'NC'] = 311  #Winterraps jetzt alle Code 311
#WW = 115
#ZR = 603
bavaria.loc[~((bavaria.NC == 600)|(bavaria.NC == 131)|(bavaria.NC == 400)|(bavaria.NC == 311)|(bavaria.NC == 115)|(bavaria.NC == 603)), 'NC'] = 1  


#%%
entries = bavaria.shape[0]
crop_types = bavaria.NC.unique()
bands = 16
data_points = 14

b = np.array(bavaria)
cnt = np.zeros((7))
# for n in range(entries):
#     cnt[b[n,3]-1] += 1

data = {}
for n in range(7):
    data[n] = np.zeros((bands, 300, data_points))


for m in range(7):
    cnt = 0
    fcnt = 0
    for n in range(entries):
        if(b[n,3]-1==m):
            # print(m)
            if (cnt==data_points):
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
inp = torch.zeros((7,bands,300,data_points)).double()
target = torch.zeros((7,300))
for n in range(7):
    inp[n,:,:,0:data_points] = torch.tensor(data[n][:,:,0:data_points], dtype=float)
    target[n,:] = n
inp = torch.einsum('abcd->acbd', inp)


### TATSAECHLICHES DATENFORMAT 
# input: 2100,bands,data_points
# target: 2100,7

inp_a = torch.zeros((2100,bands,data_points))
target_a = torch.zeros((2100,7))
k = 0
for n in range(7):
    for m in range(300):
        inp_a[k,:,:] = inp[n,m]
        target_a[k,n] = 1
        k += 1

# Transponierte, weil...
inp_a = torch.einsum('abc->acb',inp_a)

#%%
# Daten als Dataset fuer den Dataloader - ich hab das so gelernt/gelesen und daher ...
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
loader = DataLoader(set, batch_size=batch_size, shuffle=True)

#%%
from modules import Hopfield, HopfieldPooling, HopfieldLayer

class M1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hopfield = Hopfield(
            scaling=1e0,
            input_bias=True,
            input_size=16,                           # R (dimension of input - not length of input)
            output_size=128,                           # W_V_2
            )
        self.Linear_m = nn.Linear(in_features=14, out_features=1)
        self.Linear_b = nn.Linear(in_features=128, out_features=7)
        self.softmax = torch.nn.Softmax(dim=-1)
    
    
    def forward(self, sample):
        x = self.hopfield(sample)           # [128,14] neuer feature vector fuer die Zeitreihe
        x = torch.einsum('abc->acb', x)     # Transponierte fuer den linear
        y = self.Linear_m(x)                # [128,14] -> [128,1]
        y = torch.einsum('abc->acb', y)     # Transponierte fuer den linear
        z = self.Linear_b(y)                # [128,1] -> [7,1]
        return self.softmax(z)              # Wahrscheinlichkeit fuer die 7 Typen

model = M1()
model.cuda(gpu_id)
criterion = nn.BCELoss()    # gut fuer Wahrscheinlichkeiten
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-6)#, weight_decay=1e-6)


#%%
epochs = 1
for ep in range(epochs):
    print(ep)
    cor = 0
    for idx, data in enumerate(loader):
        
        sample = data[0]
        target = data[1]
        out = model(sample)
        # Fuer das erste Element im Batch ... also eine echte accuracy ist das nicht
        # ...ist eine Abschaetzung wie viele Samples waehrend des Trainings richtig sind
        if (torch.argmax(out[0]) == torch.argmax(target[0])):
            cor += 1

        # Update vom Netzwerk
        loss = criterion(out, target[:,None])
        loss.backward()
        optimizer.step()

    # Die Trainings accuracy ausgeben
    # ... batch-size = 10 also gibts 210 batches und von jedem batch wird das erste Element
    # ueberprueft ob es stimmt. N von 2100/10 richtig...
    if (ep%5==0):
        print('----', ep, '----')
        print(cor, '/', 2100/batch_size)



