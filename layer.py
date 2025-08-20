import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, softmax


class HypergraphConv(MessagePassing):
    def __init__(
            self,
            in_channels_x: int,
            in_channels_e: int,
            out_channels: int,
            use_attention: bool = False,
            attention_mode: str = 'node',
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0,
            bias: bool = True,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        assert attention_mode in ['node', 'edge']
        self.in_channels_x = in_channels_x
        self.in_channels_e = in_channels_e
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.attention_mode = attention_mode
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_x = Linear(in_channels_x, heads * out_channels, bias=False, weight_initializer='glorot')

        if use_attention:
            self.lin_e = Linear(in_channels_e, heads * out_channels, bias=False, weight_initializer='glorot')
            self.att = Parameter(torch.empty(1, heads, 2 * out_channels))
        else:
            self.lin_e = None

        # Bias
        bias_dim = heads * out_channels if concat else out_channels
        self.bias = Parameter(torch.empty(bias_dim)) if bias else None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_x.reset_parameters()
        if self.lin_e is not None:
            self.lin_e.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None,
                num_edges: Optional[int] = None) -> Tensor:

        num_nodes = x.size(0)
        if num_edges is None:
            num_edges = int(hyperedge_index[1].max()) + 1 if hyperedge_index.numel() > 0 else 0
        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x_transformed = self.lin_x(x)  # [N, heads * out_channels]

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None, "hyperedge_attr is required when use_attention=True"
            hyperedge_attr = self.lin_e(hyperedge_attr)  # [M, heads * out_channels]
            x_view = x_transformed.view(-1, self.heads, self.out_channels)
            e_view = hyperedge_attr.view(-1, self.heads, self.out_channels)
            x_i = x_view[hyperedge_index[0]]
            x_j = e_view[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            if self.attention_mode == 'node':
                alpha = softmax(alpha, hyperedge_index[1], num_nodes=num_edges)
            else:
                alpha = softmax(alpha, hyperedge_index[0], num_nodes=num_nodes)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # D_node[v] = sum_{e} H_{ve} * w_e
        D_node = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                         dim=0, dim_size=num_nodes, reduce='sum')
        D_node = D_node.clamp(min=1)
        D_half = D_node.pow(-0.5)  # D^{-1/2}

        # B_edge[e] = sum_{v} H_{ve}
        B_edge = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                         dim=0, dim_size=num_edges, reduce='sum')
        B_edge = B_edge.clamp(min=1)
        B_inv = 1.0 / B_edge

        x_transformed_scaled = D_half.unsqueeze(-1) * x_transformed  # [N, C]

        B_inv_j = B_inv[hyperedge_index[1]]  # [E]

        out = self.propagate(
            hyperedge_index,
            x=x_transformed_scaled,
            norm_weight=B_inv_j,
            alpha=alpha,
            size=(num_nodes, num_edges)
        )

        out = self.propagate(
            hyperedge_index.flip([0]),
            x=out,
            norm_weight=None,
            alpha=alpha,
            size=(num_edges, num_nodes)
        )

        out = D_half.unsqueeze(-1) * out

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
            residual = x_transformed
        else:
            out = out.mean(dim=1)
            residual = x_transformed.view(-1, self.heads, self.out_channels).mean(dim=1)

        out = out + residual

        if self.bias is not None:
            out = out + self.bias

        out = F.relu(out)
        return out

    def message(self, x_j, norm_weight=None, alpha=None):
        H, C = self.heads, self.out_channels
        out = x_j.view(-1, H, C)
        if norm_weight is not None:
            out = norm_weight.view(-1, 1, 1) * out
        if alpha is not None:
            out = alpha.view(-1, H, 1) * out
        return out.view(-1, H * C)  # [E, H*C]


class TemporalAttentionLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 n_heads,
                 num_time_steps,
                 attn_drop,
                 residual):
        super(TemporalAttentionLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual
        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))

        self.lin = nn.Linear(input_dim, input_dim, bias=True)

        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()

    def forward(self, inputs):

        position_inputs = torch.arange(0, self.num_time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(
            inputs.device)
        temporal_inputs = inputs + self.position_embeddings[position_inputs]

        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2], [0]))
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2], [0]))
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2], [0]))

        split_size = int(q.shape[-1] / self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0)
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0)
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0)

        outputs = torch.matmul(q_, k_.permute(0, 2, 1))
        outputs = outputs / (self.num_time_steps ** 0.5)

        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1)
        padding = torch.ones_like(masks) * (-2 ** 32 + 1)
        outputs_pos = torch.where(masks == 0, padding, outputs)
        outputs_neg = torch.where(masks == 0, padding, -1 * outputs)
        outputs_pos = F.softmax(outputs_pos, dim=2)
        outputs_neg = F.softmax(outputs_neg, dim=2)

        if self.training:
            outputs_pos = self.attn_dp(outputs_pos)
            outputs_neg = self.attn_dp(outputs_neg)
        outputs_pos = torch.matmul(outputs_pos, v_)
        outputs_neg = torch.matmul(outputs_neg, v_)
        outputs_pos = torch.cat(
            torch.split(outputs_pos, split_size_or_sections=int(outputs_pos.shape[0] / self.n_heads), dim=0),
            dim=2)
        outputs_neg = torch.cat(
            torch.split(outputs_neg, split_size_or_sections=int(outputs_neg.shape[0] / self.n_heads), dim=0),
            dim=2)
        outputs = (outputs_pos + outputs_neg) / 2

        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs

    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)
