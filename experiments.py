#%%
import pdb
import torch
import pandas as pd
from torch import nn
from collections import Counter
import random
import itertools

import matplotlib.pyplot as plt

from tqdm import tqdm
# %%
df = pd.read_csv('./Data/Mastri/LM2-4LUC.csv')
# %%
max(Counter(df["ID"]).values())
# %%
import torch
from torch.utils.data import Dataset, DataLoader

import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe):
        self.data = []
        for _, group in dataframe.groupby('ID'):
            # Interleave 'Time' and 'Observation' and convert to a flat tensor
            interleaved = torch.tensor(group[['Time', 'Observation']].values.flatten(), dtype=torch.float32)
            self.data.append(interleaved)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Padding is done to batch the data (i.e. ensure all sequences in a batch have the same length)
# Custom collate function for padding and attention masks
def collate_fn(batch):
    # Find the longest sequence in the batch
    max_length = max([s.size(0) for s in batch])

    # Pad sequences and create masks
    padded_sequences = []
    attention_masks = []
    for sequence in batch:
        # Calculate the number of padding elements
        padding_length = max_length - sequence.size(0)

        # Pad the sequence and create its attention mask
        padded_sequence = torch.cat([sequence, torch.zeros(padding_length, dtype=torch.float32)])
        attention_mask = torch.cat([torch.ones(sequence.size(0)), torch.zeros(padding_length)])

        padded_sequences.append(padded_sequence)
        attention_masks.append(attention_mask)

    # Stack all sequences and masks
    padded_sequences = torch.stack(padded_sequences)
    attention_masks = torch.stack(attention_masks)

    return padded_sequences, attention_masks



#%%
random.seed(0)
# create validation and training data
val_inds = random.sample(range(66),6)
train_df = df[~df['ID'].isin(val_inds)].copy()
val_df = df[df['ID'].isin(val_inds)].copy()


#%%

# Normalize data
max_time = max(train_df.Time)
max_obs = max(train_df.Observation)

val_df['Time']/=max_time
val_df['Observation']/=max_obs
train_df['Time']/=max_time
train_df['Observation']/=max_obs
#%%
train_dataset = TimeSeriesDataset(train_df)
val_dataset = TimeSeriesDataset(val_df)
#%%
train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, collate_fn=collate_fn)
#%%
batch,masks = next(iter(train_loader))
#%%
batch.unfold
#%%

## Transformer
torch.random.manual_seed(1)
mdl = nn.TransformerEncoder(nn.TransformerEncoderLayer(32,1,64,activation='gelu',dropout=0.05),4)
in_linear = nn.Linear(1,32)
out_linear = nn.Linear(32,1)
opt = torch.optim.AdamW(itertools.chain(in_linear.parameters(),mdl.parameters(),out_linear.parameters()),weight_decay=0.005)

#%%
for epoch in range(2):
    mdl = mdl.train()
    for x, masks in train_loader:
        input = x
        masks=~masks.bool()
        x = in_linear(x.unsqueeze(2)).permute(1,0,2)
        x = mdl(x,mask=nn.Transformer.generate_square_subsequent_mask(input.shape[1]),src_key_padding_mask=masks,is_causal=True)
        x = out_linear(x).permute(1,0,2).squeeze()
        loss = torch.abs(x[:,::2]-input[:,1::2])
        loss = loss[~masks[:,::2]].mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    mdl = mdl.eval()
    with torch.no_grad():
        for x, masks in val_loader:
            input = x
            masks=~masks.bool()
            x = in_linear(x.unsqueeze(2)).permute(1,0,2)
            x = mdl(x,mask=nn.Transformer.generate_square_subsequent_mask(input.shape[1]),src_key_padding_mask=masks,is_causal=True)
            x = out_linear(x).permute(1,0,2).squeeze()
            masks[:,0]=True
            diffs = torch.abs(x[:,::2]*max_obs-input[:,1::2]*max_obs)
            loss = diffs[~masks[:,::2]].mean()

print("Transformer Val Loss: ", loss.item())
# %%

# ANN
torch.random.manual_seed(1)
mdl =nn.Sequential(nn.Linear(3,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,1))
opt = torch.optim.AdamW(itertools.chain(mdl.parameters()),weight_decay=0.005)

#%%
#ANN
for epoch in range(1000):
    mdl = mdl.train()
    for x, masks in train_loader:
        input = x
        masks=masks.bool()
        x = x.unfold(1,3,2)
        x =  mdl(x)
        loss = torch.abs(x.squeeze()-input[:,3::2])
        loss = loss[masks[:,::2][:,1:]].mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
   # print(loss.item())
    mdl = mdl.eval()
    with torch.no_grad():
        for x, masks in val_loader:
            input = x
            masks=masks.bool()
            x = x.unfold(1,3,2)
            x =  mdl(x)
            
            loss = torch.abs(x.squeeze()*max_obs-input[:,3::2]*max_obs)
            loss = loss[masks[:,::2][:,1:]].mean()


fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter((input[:,2::2]*max_time)[0][:4],(x.squeeze()*max_obs)[0][:4], label='Prediction')
plt.scatter((input[:,2::2]*max_time)[0][:4],(input[:,3::2]*max_obs)[0][:4], label='Expectation')
plt.xlabel('Time')
plt.ylabel('Observation')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter((input[:,2::2]*max_time)[1],(x.squeeze()*max_obs)[1], label='Prediction')
plt.scatter((input[:,2::2]*max_time)[1],(input[:,3::2]*max_obs)[1], label='Expectation')
plt.xlabel('Time')
plt.ylabel('Observation')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter((input[:,2::2]*max_time)[2],(x.squeeze()*max_obs)[2], label='Prediction')
plt.scatter((input[:,2::2]*max_time)[2],(input[:,3::2]*max_obs)[2], label='Expectation')
plt.xlabel('Time')
plt.ylabel('Observation')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter((input[:,2::2]*max_time)[3][:6],(x.squeeze()*max_obs)[3][:6], label='Prediction')
plt.scatter((input[:,2::2]*max_time)[3][:6],(input[:,3::2]*max_obs)[3][:6], label='Expectation')
plt.xlabel('Time')
plt.ylabel('Observation')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter((input[:,2::2]*max_time)[4],(x.squeeze()*max_obs)[4], label='Prediction')
plt.scatter((input[:,2::2]*max_time)[4],(input[:,3::2]*max_obs)[4], label='Expectation')
plt.xlabel('Time')
plt.ylabel('Observation')
plt.legend()
plt.show()


print("ANN Val Loss: ", loss.item())

## Linear
torch.random.manual_seed(1)
mdl =nn.Sequential(nn.Linear(3,1))
opt = torch.optim.AdamW(itertools.chain(mdl.parameters()),weight_decay=0.005)

#%%
#Linear
for epoch in range(10000):
    mdl = mdl.train()
    for x, masks in train_loader:
        input = x
        masks=masks.bool()
        x = x.unfold(1,3,2)
        x =  mdl(x)
        loss = torch.abs(x.squeeze()-input[:,3::2])
        loss = loss[masks[:,::2][:,1:]].mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
   # print(loss.item())
    mdl = mdl.eval()
    with torch.no_grad():
        for x, masks in val_loader:
            input = x
            masks=masks.bool()
            x = x.unfold(1,3,2)
            x =  mdl(x)
            loss = torch.abs(x.squeeze()*max_obs-input[:,3::2]*max_obs)
            loss = loss[masks[:,::2][:,1:]].mean()

print("Linear Model Val Loss: ", loss.item())

# %%

# Visualize the data
fig, ax = plt.subplots(figsize=(10, 5))
plt.scatter(train_df['Time'], train_df['Observation'], label='Train')
plt.scatter(val_df['Time'], val_df['Observation'], label='Validation')
plt.xlabel('Time')
plt.ylabel('Observation')
plt.legend()
plt.show()



