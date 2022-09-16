import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
import argparse


class GNN_LBA(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GNN_LBA, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.conv3 = GCNConv(hidden_dim * 2, hidden_dim * 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim * 4)
        self.conv4 = GCNConv(hidden_dim * 4, hidden_dim * 4)
        self.bn4 = nn.BatchNorm1d(hidden_dim * 4)
        self.conv5 = GCNConv(hidden_dim * 4, hidden_dim * 8)
        self.bn5 = nn.BatchNorm1d(hidden_dim * 8)
        self.fc1 = nn.Linear(hidden_dim * 8, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, 1)

    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.conv4(x, edge_index, edge_weight)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x, edge_index, edge_weight)
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.25, training=self.training)
        return self.fc2(x).view(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--seqid', type=int, default=30)
    parser.add_argument('--precomputed', action='store_true')
    args = parser.parse_args()

    model = GNN_LBA(3, hidden_dim=args.hidden_dim)
    print(sum(p.numel() for p in model.parameters()) )
