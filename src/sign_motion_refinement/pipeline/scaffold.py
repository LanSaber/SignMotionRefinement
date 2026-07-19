from __future__ import annotations

import torch

from sign_motion_refinement.models.baselines import build_baseline


def make_uniform_tau(length, device=None, dtype=torch.float32):
    length = int(length)
    if length <= 1:
        return torch.zeros(length, device=device, dtype=dtype)
    return torch.linspace(0.0, 1.0, length, device=device, dtype=dtype)


def anchor_indices(length, stride=4):
    length = int(length)
    if length <= 1:
        return torch.zeros(1, dtype=torch.long)
    stride = max(int(stride), 1)
    anchors = list(range(0, length, stride))
    if anchors[-1] != length - 1:
        anchors.append(length - 1)
    return torch.tensor(sorted(set(anchors)), dtype=torch.long)


def build_sequence_scaffold(x, length, stride=4, kind="slerp"):
    length = int(length)
    if length <= 0:
        return x.new_zeros((0, x.shape[-1])), torch.zeros(
            0, dtype=torch.bool, device=x.device
        )
    if length <= 1:
        return x[:length].clone(), torch.ones(length, dtype=torch.bool, device=x.device)

    tau = make_uniform_tau(length, device=x.device, dtype=x.dtype)
    anchors = anchor_indices(length, stride=stride).to(x.device)
    anchor_mask = torch.zeros(length, dtype=torch.bool, device=x.device)
    anchor_mask[anchors] = True

    baseline = build_baseline(kind)
    baseline.fit(tau[anchors].detach().cpu(), x[:length][anchors].detach().cpu())
    scaffold = baseline.predict(tau, device=x.device, dtype=x.dtype)
    return scaffold, anchor_mask


def build_batch_scaffold(x, lengths, stride=4, kind="slerp"):
    scaffold = torch.zeros_like(x)
    anchor_mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device)
    for idx, length in enumerate(lengths.detach().cpu().tolist()):
        seq_scaffold, seq_anchor_mask = build_sequence_scaffold(
            x[idx, :length],
            length,
            stride=stride,
            kind=kind,
        )
        scaffold[idx, :length] = seq_scaffold
        anchor_mask[idx, :length] = seq_anchor_mask
    return scaffold, anchor_mask


def normalized_time_grid(lengths, max_len=None, device=None, dtype=torch.float32):
    batch = len(lengths)
    max_len = int(max_len if max_len is not None else int(lengths.max().item()))
    tau = torch.zeros(batch, max_len, 1, device=device, dtype=dtype)
    for idx, length in enumerate(lengths.detach().cpu().tolist()):
        tau[idx, :length, 0] = make_uniform_tau(length, device=device, dtype=dtype)
    return tau
