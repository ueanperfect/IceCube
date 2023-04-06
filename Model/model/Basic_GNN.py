import torch.nn as nn
from typing import List
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch
from ..utils import calculate_distance_matrix


class EuclideanGraphBuilder(nn.Module):
    """Builds graph according to Euclidean distance between nodes.
    See https://arxiv.org/pdf/1809.06166.pdf.
    """

    def __init__(
            self,
            sigma: float,
            threshold: float = 0.0,
            columns: List[int] = None,
    ):
        """Construct `EuclideanGraphBuilder`."""
        # Base class constructor
        super().__init__()

        # Check(s)
        if columns is None:
            columns = [0, 1, 2]

        # Member variable(s)
        self._sigma = sigma
        self._threshold = threshold
        self._columns = columns

    def forward(self, data: Data) -> Data:
        """Forward pass."""
        # Constructs the adjacency matrix from the raw, DOM-level data and
        # returns this matrix
        xyz_coords = data.x[:, self._columns]

        # Construct block-diagonal matrix indicating whether pulses belong to
        # the same event in the batch
        batch_mask = data.batch.unsqueeze(dim=0) == data.batch.unsqueeze(dim=1)

        distance_matrix = calculate_distance_matrix(xyz_coords)
        affinity_matrix = torch.exp(
            -0.5 * distance_matrix ** 2 / self._sigma ** 2
        )

        # Use softmax to normalise all adjacencies to one for each node
        exp_row_sums = torch.exp(affinity_matrix).sum(axis=1)
        weighted_adj_matrix = torch.exp(
            affinity_matrix
        ) / exp_row_sums.unsqueeze(dim=1)

        # Only include edges with weights that exceed the chosen threshold (and
        # are part of the same event)
        sources, targets = torch.where(
            (weighted_adj_matrix > self._threshold) & (batch_mask)
        )
        edge_weights = weighted_adj_matrix[sources, targets]

        data.edge_index = torch.stack((sources, targets))
        data.edge_weight = edge_weights

        return data


class DenseDynBlock(nn.Module):
    """
    Dense Dynamic graph convolution block
    """

    def __init__(self, in_channels, out_channels=64, sigma=0.5):
        super(DenseDynBlock, self).__init__()
        self.GraphBuilder = EuclideanGraphBuilder(sigma=sigma)
        self.gnn = SAGEConv(in_channels, out_channels)

    def forward(self, data):
        data1 = self.GraphBuilder(data)
        x, edge_index, batch = data1.x, data1.edge_index, data1.batch
        x = self.gnn(x, edge_index)
        data1.x = torch.cat((x, data.x), 1)
        return data1


class MyGNN(nn.Module):
    """
    Dynamic graph convolution layer
    """

    def __init__(self, in_channels, hidden_channels, out_channels, n_blocks):
        super().__init__()
        self.n_blocks = n_blocks
        self.head = SAGEConv(in_channels, hidden_channels)
        c_growth = hidden_channels
        self.gnn = nn.Sequential(
            *[DenseDynBlock(hidden_channels + i * c_growth, c_growth) for i in range(n_blocks - 1)])
        fusion_dims = int(
            hidden_channels * self.n_blocks + c_growth * ((1 + self.n_blocks - 1) * (self.n_blocks - 1) / 2))
        self.linear = nn.Linear(fusion_dims, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        data.x = self.head(x, edge_index)
        feats = [data.x]
        for i in range(self.n_blocks - 1):
            data = self.gnn[i](data)
            feats.append(data.x)
        feats = torch.cat(feats, 1)
        x = pyg_nn.global_mean_pool(feats, data.batch)
        out = F.relu(self.linear(x))
        # out = F.sigmoid(out)
        # out[:, 0] = out[:, 0] * 2 * torch.pi
        # out[:, 1] = out[:, 1] * torch.pi
        return out
