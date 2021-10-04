#%%
import torch
import numpy as np 
from modules import Hopfield, HopfieldPooling, HopfieldLayer
from torch.nn.functional import mse_loss


#%% DATA 1 (WORKING!)
tot = 7
originals = torch.zeros((tot,40))
for n in range(tot):
    originals[n] = torch.rand(40)

data = torch.zeros((tot,40)) + 0

for n in range(tot):
    # rd = int(torch.rand(1) * 3)
    originals[n] = originals[n] #/torch.sum(originals[n])
    data[n,0:40] = originals[n,0:40] + torch.rand(40)/10

print(data[None,None,0].shape)
print(originals[None].shape)

#%% MODEL
hopfield = Hopfield(
    scaling=1e6,
    input_bias=False,

    # do not project layer input
    state_pattern_as_static=True,
    stored_pattern_as_static=True,
    pattern_projection_as_static=True,

    # do not pre-process layer input
    normalize_stored_pattern=False,
    normalize_stored_pattern_affine=False,
    normalize_state_pattern=False,
    normalize_state_pattern_affine=False,
    normalize_pattern_projection=False,
    normalize_pattern_projection_affine=False,

    # do not post-process layer output
    disable_out_projection=True)

n = 6
print(originals[None].shape, data[None,None,n].shape)
print(n)
out = hopfield((originals[None], data[None,None,n], originals[None]))
print('INPUT:')
print(data[None,None,n])
print('WANTED:')
print(originals[None,None,n])
print('OUTPUT:')
print(out)

#%%
import matplotlib.pyplot as plt
_, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
ax[0].plot(torch.linspace(0,40,40), originals[0,:])
ax[0].plot(torch.linspace(0,40,40), originals[1,:])

### ----------------------------------------------- ###
### ----------------------------------------------- ###
### ----------------------------------------------- ###
#%% DATA 2 (NOT WORKING!)
import matplotlib.pyplot as plt

unique = 10
tot = 100
originals = torch.zeros((unique,tot,40))
prototype = torch.zeros((unique,40))
for i in range(unique):
    for n in range(tot):
        originals[i,n] = i*torch.linspace(0,40,40)/(40*unique) + torch.rand(40)/10
    prototype[i,:] = torch.sum(originals[i], 0)/tot

print(originals[0,0,None,None].shape)
print(prototype[None].shape)

#%%
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
for n in range(10):
    ax.plot(torch.linspace(0,40,40), originals[n,0,:])#, label=f'sample of type {n}')
    ax.plot(torch.linspace(0,40,40), prototype[n,:], linestyle='dashed', label=f'prototype {n}')
    ax.legend()

#%%
hopfield = Hopfield(
    input_size=40,
    scaling=1e6,
    input_bias=True,

    # do not project layer input
    state_pattern_as_static=True,
    stored_pattern_as_static=True,
    pattern_projection_as_static=True,

    # do not pre-process layer input
    normalize_stored_pattern=False,
    normalize_stored_pattern_affine=False,
    normalize_state_pattern=False,
    normalize_state_pattern_affine=False,
    normalize_pattern_projection=False,
    normalize_pattern_projection_affine=False,

    # do not post-process layer output
    disable_out_projection=True)

#%%
pattern = 1
sample = 10
out = hopfield((prototype[None], originals[pattern,sample,None,None], prototype[None]))

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
ax.plot(torch.linspace(0,40,40), originals[pattern,sample,:], label=f'Input Type {pattern}')
ax.plot(torch.linspace(0,40,40), prototype[pattern,:], label=f'Prototype {pattern}')
ax.plot(torch.linspace(0,40,40), out[0,0,:].detach(), label='Output')
ax.legend()

### ----------------------------------------------- ###
### ----------------------------------------------- ###
### ----------------------------------------------- ###
#%% DATA 3 (WORKING!)
import matplotlib.pyplot as plt

unique = 10
tot = 100
originals = torch.zeros((unique,tot,40))
prototype = torch.zeros((unique,40))
for i in range(unique):
    for n in range(tot):
        originals[i,n] = torch.sin(i*torch.linspace(0,2*np.pi,40)) + (torch.rand(40)-0.5)*2
    prototype[i,:] = torch.sum(originals[i], 0)/tot

print(originals[0,0,None,None].shape)
print(prototype[None].shape)

#%%
_, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
for n in range(3):
    ax.plot(torch.linspace(0,2*np.pi,40), originals[n,0,:])#, label=f'sample of type {n}')
    ax.plot(torch.linspace(0,2*np.pi,40), prototype[n,:], linestyle='dashed', label=f'prototype {n}')
    ax.legend()

#%%
hopfield = Hopfield(
    input_size=40,
    scaling=1e6,
    input_bias=True,

    # do not project layer input
    state_pattern_as_static=True,
    stored_pattern_as_static=True,
    pattern_projection_as_static=True,

    # do not pre-process layer input
    normalize_stored_pattern=False,
    normalize_stored_pattern_affine=False,
    normalize_state_pattern=False,
    normalize_state_pattern_affine=False,
    normalize_pattern_projection=False,
    normalize_pattern_projection_affine=False,

    # do not post-process layer output
    disable_out_projection=True)

#%%
pattern = 9
sample = 11
out = hopfield((prototype[None], originals[pattern,sample,None,None], prototype[None]))

_, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
ax.plot(torch.linspace(0,2*np.pi,40), originals[pattern,sample,:], label=f'Input Type {pattern}')
ax.plot(torch.linspace(0,2*np.pi,40), prototype[pattern,:], label=f'Prototype {pattern}')
ax.plot(torch.linspace(0,2*np.pi,40), out[0,0,:].detach(), linestyle='dotted', label='Output')
ax.legend()

# %%
