"""Implementation of Relational graph convolution layer"""
import torch
from torch import nn

from dgl import function as fn
from dgl.nn.pytorch.linear import TypedLinear


class RelGraphConv(nn.Module):
  

    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        regularizer=None,
        num_bases=None,
        bias=True,
        activation=None,
        self_loop=True,
        dropout=0.0,
        layer_norm=False,
    ):
        super().__init__()
        if regularizer is not None and num_bases is None:
            num_bases = num_rels
        self.linear_r = TypedLinear(
            in_feat, out_feat, num_rels, regularizer, num_bases
        )
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(
                out_feat, elementwise_affine=True
            )

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )

        self.dropout = nn.Dropout(dropout)

    def message(self, edges):
        """Message function."""
        m = self.linear_r(edges.src["h"], edges.data["etype"], self.presorted)
        if "norm" in edges.data:
            m = m * edges.data["norm"]
        return {"m": m}

    def forward(self, g, feat, etypes, norm=None, *, presorted=False):
        """Forward computation."""
        self.presorted = presorted
        with g.local_scope():
            g.srcdata["h"] = feat
            if norm is not None:
                g.edata["norm"] = norm
            g.edata["etype"] = etypes
            # message passing
            g.update_all(self.message, fn.sum("m", "h"))
            # apply bias and activation
            h = g.dstdata["h"]
            if self.layer_norm:
                h = self.layer_norm_weight(h)
            if self.bias:
                h = h + self.h_bias
            if self.self_loop:
                h = h + feat[: g.num_dst_nodes()] @ self.loop_weight
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)
            return h
