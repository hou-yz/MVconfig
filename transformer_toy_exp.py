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
        self.steps = torch.randint(0, num_cam - 1, [length])
        self.data = torch.randn([length, num_cam, config_dim])
        self.data[torch.arange(num_cam).repeat(length, 1) > self.steps[:, None]] = CONFIGS_PADDING_VALUE

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
            self.transformer = nn.TransformerEncoderLayer(hidden_dim, 8, hidden_dim * 4, batch_first=True)
            self.state_token = nn.Parameter(torch.randn(hidden_dim))
        self.output_head = nn.Linear(hidden_dim, num_cam)

    def forward(self, configs, step):
        B, N, C = configs.shape
        if self.arch == 'transformer':
            x_config = self.config_branch(configs.flatten(0, 1)).unflatten(0, [B, N])
            x = x_config + self.positional_embedding.to(configs.device)
            x = torch.cat([self.state_token.repeat([B, 1, 1]), x], dim=1)
            mask = torch.arange(-1, N).repeat([B, 1]) > step[:, None]
            x = self.transformer(x, src_key_padding_mask=mask)
            # Classifier "token" as used by standard language architectures
            x = x[:, 0]
        else:
            x = x_config = self.config_branch(configs.flatten(1, 2))
        out = self.output_head(x)
        return out


if __name__ == '__main__':
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
    # fc: 14.18504 -> 0.05645
    # transformer v0 (encoder + prefix multi-query): 6.12997 -> 0.02429
    # transformer v0.1 (encoder + prefix multi-query + mask): 6.94008 -> 0.02217
    # transformer v1 (encoder + prefix same query): 15.10486 -> 0.17619
    # transformer v1.1 (encoder + prefix same query + mask): 13.36482 -> 0.22661
