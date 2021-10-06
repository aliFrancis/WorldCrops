#%%
import torch

### WORDS
y = torch.rand(5,3)
yt = torch.einsum('ab->ba',y)

### HOPFIELD MATRICES
Wq = torch.rand(3,3)
Wk = torch.rand(3,3)

### SVD AND APPROXIMATION
u1,s1,v1 = torch.svd(Wq)
v1 = v1.transpose(-2, -1).conj()
ss1 = torch.eye(u1.shape[1],v1.shape[0])*s1
u2,s2,v2 = torch.svd(Wk)
v2 = v2.transpose(-2, -1).conj()
ss2 = torch.eye(u2.shape[1],v2.shape[0])*s2

ss1_a = torch.zeros(3,3)
ss2_a = torch.zeros(3,3)
ss1_a[0,0] = ss1[0,0]
ss1_a[1,1] = ss1[1,1]
ss2_a[0,0] = ss2[0,0]
ss2_a[1,1] = ss2[1,1]


print(y@u1)
print(' ')
print(ss1_a@v1@u2@ss2_a)
print(' ')
print(v2@yt)
# print(y@u1@ss1_a@v1@u2@ss2_a)
print(' ')
print(' ')
print(ss1_a@v1@u2@ss2_a@v2@yt)


# %%
