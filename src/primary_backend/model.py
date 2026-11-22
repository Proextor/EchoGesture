"""Tiny depthwise-separable temporal CNN over landmark sequences."""
import torch
import torch.nn as nn

from .config import MODEL_IN_DIM, T


class DSBlock(nn.Module):
    def __init__(self, ch, k=5, dilation=1, p=0.2):
        super().__init__()
        pad = (k // 2) * dilation
        self.dw = nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation, groups=ch, bias=False)
        self.pw = nn.Conv1d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return x + self.drop(self.act(self.bn(self.pw(self.dw(x)))))


class EchoNet(nn.Module):
    def __init__(self, n_classes, in_dim=MODEL_IN_DIM, ch=192, dilations=(1, 2, 4, 1, 2), p=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv1d(in_dim, ch, 1, bias=False), nn.BatchNorm1d(ch), nn.ReLU())
        self.blocks = nn.Sequential(*[DSBlock(ch, 5, d, p) for d in dilations])
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(ch * 2, n_classes)

    def forward(self, x):                 # x: [B, T, F]
        x = self.inp(x.transpose(1, 2))   # [B, C, T]
        x = self.blocks(x)
        x = torch.cat([x.mean(-1), x.amax(-1)], dim=1)   # avg + max pool over time
        return self.head(self.drop(x))


class Ensemble(nn.Module):
    """Average the logits of several EchoNets (trained with different seeds) inside one graph."""
    def __init__(self, nets):
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, x):
        out = self.nets[0](x)
        for n in self.nets[1:]:
            out = out + n(x)
        return out / len(self.nets)


def example_input(batch=1):
    return torch.zeros(batch, T, MODEL_IN_DIM)
