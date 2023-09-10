import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from src.models.mvcontrol import create_pos_embedding

CONFIGS_PADDING_VALUE = -3


class NumWordDataset(Dataset):
    def __init__(self, length, num_cam, config_dim):
        self.num_cam = num_cam
        self.steps = torch.randint(0, num_cam, [length])
        self.data = torch.randn([length, num_cam, config_dim])
        self.data[torch.arange(num_cam).repeat(length, 1) >= self.steps[:, None]] = CONFIGS_PADDING_VALUE

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, index):
        return self.data[index], self.steps[index]


class Counter(nn.Module):
    def __init__(self, num_cam, config_dim, arch='transformer'):
        super().__init__()
        hidden_dim = 128
        self.arch = arch
        self.config_branch = nn.Sequential(
            nn.Linear(config_dim * (1 if arch == 'transformer' else num_cam), hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        if arch == 'transformer':
            # transformer
            self.positional_embedding = create_pos_embedding(num_cam, hidden_dim)
            # CHECK: batch_first=True for transformer
            encoder_layer = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, 3)
            self.state_token = nn.Parameter(torch.randn(num_cam, hidden_dim))
        self.output_head = nn.Linear(hidden_dim, num_cam)

    def forward(self, configs, step):
        B, N, C = configs.shape
        if self.arch == 'transformer':
            x = self.config_branch(configs.flatten(0, 1)).unflatten(0, [B, N])
            token_location = (torch.arange(N).repeat([B, 1]) == step[:, None])
            x[token_location] = self.state_token[step]
            x += self.positional_embedding
            # mask out future steps
            # mask = (torch.arange(N).repeat([B, 1]) > step[:, None])
            # x[mask] = 0
            x = self.transformer(x)
            x = x[token_location]
        else:
            x = self.config_branch(configs.flatten(1, 2))
        out = self.output_head(x)
        return out


if __name__ == '__main__':

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    num_cam, config_dim = 4, 7
    batch_size, num_epochs = 32, 10
    trainset = NumWordDataset(360, num_cam, config_dim)
    train_loader = DataLoader(trainset, batch_size, shuffle=True)

    model = Counter(num_cam, config_dim)
    optimizer = Adam(model.parameters(), lr=3e-4)

    for epoch in range(1, num_epochs + 1):
        train_loss = 0
        for batch_idx, (configs, step) in enumerate(train_loader):
            out = model(configs, step)
            loss = F.cross_entropy(out, step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        print(f'{train_loss:.5f}')

    # loss
    # fc: 15.75898 -> 0.06424
    # transformer v0.0 (encoder + prefix + multi token): 7.04339 -> 0.03376
    # transformer v0.0.1 (encoder + prefix + multi token + mask): 6.89566 -> 0.03512
    # transformer v0.1 (encoder + prefix + same token): 13.48105 -> 0.16533
    # transformer v0.1.1 (encoder + prefix + same token + mask): 13.02086 -> 0.09835
    # transformer v1.0 (encoder + in-place + multi token): 7.02257 -> 0.03102
    # transformer v1.0.1 (encoder + in-place + multi token + mask): 7.02461 -> 0.03238
    # transformer v1.1 (encoder + in-place + same token): 11.66371 -> 0.03283
    # transformer v1.1.1 (encoder + in-place + same token + mask): 11.73027 -> 0.06264
    # transformer v2.0 (full transformer + multi token): 3.48481 -> 0.02469
    # transformer v2.0.0.5 (full transformer + multi token + put 0 to masked location): 3.77019 -> 0.02691
    # transformer v2.0.1 (full transformer + multi token + mask): 3.99520 -> 0.02789
    # transformer v2.0 (full transformer + same token): 13.46105 -> 0.04392
    # transformer v2.0.1 (full transformer + same token + mask): 11.68639 -> 0.04158
