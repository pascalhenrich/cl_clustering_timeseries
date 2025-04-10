import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np

def temporal_augmentation(x, alpha=0.8):
    jitter = torch.normal(mean=0.0, std=alpha, size=x.shape)
    return x + jitter

class TemporalEncoder(nn.Module):
    def __init__(self):
        super(TemporalEncoder, self).__init__()
        self.bilstm = nn.LSTM(1, 1, batch_first=True, bidirectional=True)
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x, _ = self.bilstm(x.view(batch_size, seq_len,1))
        x = self.layer(x).view(batch_size, seq_len)
        return x
    
class InstanceLevelTransformation(nn.Module):
    def __init__(self, input_size, output_size):
        super(InstanceLevelTransformation, self).__init__()
        self.layer1 = nn.Linear(input_size, input_size//2)
        self.layer2 = nn.Linear(input_size//2, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
class ClusterLevelTransformation(nn.Module):
    def __init__(self, input_size,cluster_size):
        super(ClusterLevelTransformation, self).__init__()
        self.layer1 = nn.Linear(input_size, input_size//4*3)
        self.layer2 = nn.Linear(input_size//4*3, input_size//2)
        self.layer3 = nn.Linear(input_size//2, input_size//4)
        self.layer4 = nn.Linear(input_size//4, cluster_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.softmax(x, dim=1)
        return x

# Model Definition
class CDCCModel(nn.Module):
    def __init__(self, seq_len=17520, feature_length=1752, cluster_size=10):
        super(CDCCModel, self).__init__()
        self.temporal_encoder = TemporalEncoder()
        self.temporal_transform = InstanceLevelTransformation(seq_len,feature_length)
        self.cluster_transform = ClusterLevelTransformation(seq_len,cluster_size)

    def forward(self, x, train=True):
        if train:
            x_t, x_t_aug = x           

            # Encodings
            h_t = self.temporal_encoder(x_t)
            h_t_aug = self.temporal_encoder(x_t_aug)

            # Transformations
            z_i = self.temporal_transform(h_t)
            z_i_aug = self.temporal_transform(h_t_aug)
            z_c = self.cluster_transform(h_t)
            z_c_aug = self.cluster_transform(h_t_aug)
            
            return z_i,z_i_aug,z_c,z_c_aug
        else:
            return torch.argmax(self.cluster_transform(self.temporal_encoder(x)),dim=1)


def get_batch(tensor, batch_size):
    indices = np.random.choice(tensor.shape[0], batch_size, replace=False)
    return tensor[indices]

class InfocNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfocNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, original, augmented):
        # Batch size
        batch_size = original.size(0)

        # Normalize original and augmented embeddings
        original = F.normalize(original, p=2, dim=1)   # (batch_size, embedding_dim)
        augmented = F.normalize(augmented, p=2, dim=1)  # (batch_size, embedding_dim)

        # Concatenate original and augmented embeddings (2 * batch_size, embedding_dim)
        embeddings = torch.cat([original, augmented], dim=0)

        # Compute similarity matrix (2 * batch_size, 2 * batch_size)
        similarity_matrix = torch.mm(embeddings, embeddings.T)

        # Create labels: For each anchor in the batch, the positive pair is the corresponding augmented one
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)])

        # Mask out self-similarity to avoid considering the same sample as both positive and negative
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(original.device)
        similarity_matrix.masked_fill_(mask, -float('inf'))

        # Scale similarity matrix by temperature
        similarity_matrix /= self.temperature

        # Compute the InfoNCE loss
        loss = F.cross_entropy(similarity_matrix, labels, reduction='mean')

        return loss

data = pd.read_csv("Final_Energy_dataset.csv")
data['Date'] = pd.to_datetime(data['Date'])
data = data[data['Date'].dt.date != pd.to_datetime('2012-02-29').date()]    
prosumption = {}
for i in range(101,301):
    if (data[f'load_{i}'].isna().sum() == 0) & (data[f'pv_{i}'].isna().sum() == 0):
        prosumption[f'prosumption_{i}'] = data[f'load_{i}'] - data[f'pv_{i}']
data_train_df = pd.DataFrame(prosumption).reset_index(drop=True).T
data_train_tensor = torch.Tensor(data_train_df.values).reshape(600,17520)
data = data[((data['Date'].dt.month >= 7) & (data['Date'].dt.year == 2010)) 
            | (((data['Date'].dt.month < 7) & (data['Date'].dt.year == 2011))) 
            | (data['Date'] == pd.to_datetime('2011-07-01 00:00:00'))]
prosumption = {}
for i in range(1,101):
    if (data[f'load_{i}'].isna().sum() == 0) & (data[f'pv_{i}'].isna().sum() == 0):
        prosumption[f'prosumption_{i}'] = data[f'load_{i}'] - data[f'pv_{i}']
data_test_df = pd.DataFrame(prosumption).reset_index(drop=True).T
data_test_tensor = torch.Tensor(data_test_df.values).reshape(100,17520)

batch_size = 16
cluster_size = 10
epochs = 50

modelA = CDCCModel(cluster_size=cluster_size)
optimizer = optim.Adam(modelA.parameters(), lr=0.001)
loss_func = InfocNCELoss()

for epoch in range(epochs):

    data = get_batch(data_train_tensor, batch_size)
    data_aug = temporal_augmentation(data)
    print(modelA(data_test_tensor[0:10],train=False))
    
   
    
    a,b,c,d = modelA((data,data_aug))
    
    loss_i = (loss_func(a,b)+loss_func(b,a))/2
    loss_c = (loss_func(c.T,d.T)+loss_func(d.T,c.T))/2

    # p_c = torch.mean(c, dim=0)
    # p_c_aug = torch.mean(d, dim=0)
    # loss_ce = -torch.sum(p_c*torch.log(p_c)-p_c_aug*torch.log(p_c_aug))
    print(loss_i,loss_c)
    loss = loss_i + loss_c 
    print(f'Epoch {epoch+1} / {epochs}, Loss: {loss.item()}')


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
   