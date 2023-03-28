import torch
import torch.nn.functional as F
from torch.nn import Linear

from torch_geometric.nn import JumpingKnowledge, SAGEConv, global_mean_pool

import argparse

from citation import get_planetoid_dataset, random_planetoid_splits, run

from torch_geometric.profile import rename_profile_file

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', action='store_true')
parser.add_argument('--runs', type=int, default=100)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--no_normalize_features', action='store_true')
parser.add_argument('--inference', action='store_true')
parser.add_argument('--profile', action='store_true')
parser.add_argument('--bf16', action='store_true')
parser.add_argument('--ckpt_path', type=str, required=True)

args = parser.parse_args()

class GraphSAGE(torch.nn.Module):
    '''
        layers = [1, 2, 3, 4, 5]
        hiddens = [16, 32, 64, 128]
    '''
    def __init__(self, dataset, num_layers=2, hidden=16):
        super().__init__()
        self.conv1 = SAGEConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GraphSAGEWithJK(torch.nn.Module):
    def __init__(self, dataset, num_layers, hidden, mode='cat'):
        super().__init__()
        self.conv1 = SAGEConv(dataset.num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, dataset.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]
        x = self.jump(xs)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

dataset = get_planetoid_dataset(args.dataset, not args.no_normalize_features)
permute_masks = random_planetoid_splits if args.random_splits else None
run(dataset, GraphSAGE(dataset), args.runs, args.epochs, args.lr, args.weight_decay,
    args.early_stopping, args.inference, args.profile, args.bf16,
    permute_masks, ckpt_path=args.ckpt_path)

if args.profile:
    rename_profile_file('citation', GCNConv.__name__, args.dataset,
                        str(args.random_splits),
                        'inference' if args.inference else 'train')