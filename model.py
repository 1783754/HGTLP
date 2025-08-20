import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import TemporalAttentionLayer, HypergraphConv


class HGTLP(torch.nn.Module):
    def __init__(self, input_dim, edge_input_dim, hidden_size, latent_dim, num_window, with_dropout, heads):
        super(HGTLP, self).__init__()
        self.latent_dim = latent_dim
        self.heads = heads

        att_kwargs = {
            'use_attention': True,
            'attention_mode': 'node',
            'heads': self.heads,
            'concat': True,
            'negative_slope': 0.2,
            'dropout': 0.1 if with_dropout else 0.0,
        }
        self.conv_for_node = nn.ModuleList()
        self.conv_for_edge = nn.ModuleList()
        self.collapsed_conv_for_node = nn.ModuleList()
        self.collapsed_conv_for_edge = nn.ModuleList()

        self.conv_for_node.append(
            HypergraphConv(input_dim, edge_input_dim, latent_dim[0] * 2, **att_kwargs)
        )
        for i in range(1, len(latent_dim)):
            self.conv_for_node.append(
                HypergraphConv(self.heads * latent_dim[i - 1] * 2, self.heads * latent_dim[i - 1], latent_dim[i] * 2,
                               **att_kwargs)
            )

        self.conv_for_edge.append(
            HypergraphConv(edge_input_dim, input_dim, latent_dim[0], **att_kwargs)
        )
        for i in range(1, len(latent_dim)):
            self.conv_for_edge.append(
                HypergraphConv(self.heads * latent_dim[i - 1], self.heads * latent_dim[i - 1] * 2, latent_dim[i],
                               **att_kwargs)
            )
        self.collapsed_conv_for_node.append(
            HypergraphConv(input_dim, edge_input_dim, latent_dim[0] * 2, **att_kwargs)
        )
        for i in range(1, len(latent_dim)):
            self.collapsed_conv_for_node.append(
                HypergraphConv(self.heads * latent_dim[i - 1] * 2, self.heads * latent_dim[i - 1], latent_dim[i] * 2,
                               **att_kwargs)
            )

        self.collapsed_conv_for_edge.append(
            HypergraphConv(edge_input_dim, input_dim, latent_dim[0], **att_kwargs)
        )
        for i in range(1, len(latent_dim)):
            self.collapsed_conv_for_edge.append(
                HypergraphConv(self.heads * latent_dim[i - 1], self.heads * latent_dim[i - 1] * 2, latent_dim[i],
                               **att_kwargs)
            )

        self.temporal_dim = 5 * self.heads * sum(latent_dim)
        self.temporal_layer = TemporalAttentionLayer(self.temporal_dim,
                                                     1,
                                                     num_window,
                                                     attn_drop=0.5,
                                                     residual=True)

        self.linear1 = nn.Linear(self.temporal_dim * 2, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 2)
        self.with_dropout = with_dropout

    def forward(self, graph_list):

        device = next(self.parameters()).device

        time_series = graph_list[:-1]

        structural_outs = [
            self.encode_spatial_features(g.to(device), self.conv_for_node, self.conv_for_edge)
            for g in time_series
        ]
        structural_outs = [x.unsqueeze(1) for x in structural_outs]
        structural_outs = torch.cat(structural_outs, dim=1)

        collapsed_g = graph_list[-1].to(device)
        collapsed_out = self.encode_spatial_features(
            collapsed_g,
            self.collapsed_conv_for_node,
            self.collapsed_conv_for_edge
        )

        temporal_outs = self.temporal_layer(structural_outs)[:, -1, :]  # [N, D]

        output = torch.cat([temporal_outs, collapsed_out], dim=1)

        output = self.linear1(output)
        output = F.relu(output)
        if self.with_dropout:
            output = F.dropout(output, training=self.training)
        output = self.linear2(output)
        output = F.log_softmax(output, dim=1)

        return output

    def encode_spatial_features(self, g, conv_for_node, conv_for_edge):

        x, edge_x, edge_index = g.x, g.edge_attr, g.edge_index
        marks, edge_marks = g.marks, g.edge_marks

        all_x = []
        all_edge_x = []

        for lv in range(len(self.latent_dim)):
            x1 = conv_for_node[lv](
                x=x,
                hyperedge_index=edge_index,
                hyperedge_attr=edge_x
            )
            all_x.append(x1)

            edge_x1 = conv_for_edge[lv](
                x=edge_x,
                hyperedge_index=edge_index.flip([0]),
                hyperedge_attr=x
            )
            all_edge_x.append(edge_x1)
            x, edge_x = x1, edge_x1
        x_out = torch.cat(all_x, dim=1)
        edge_x_out = torch.cat(all_edge_x, dim=1)
        e = edge_marks
        u = edge_index[0, e]
        v = edge_index[1, e]
        x_u = x_out[u]
        x_v = x_out[v]
        # min/max
        node_aggr = torch.cat([
            torch.min(x_u, x_v),
            torch.max(x_u, x_v)
        ], dim=1)
        edge_rep = edge_x_out[edge_marks]
        # node_aggr + edge_rep
        structural_out = torch.cat([node_aggr, edge_rep], dim=1)

        return structural_out
