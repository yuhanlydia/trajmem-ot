from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .core import MemoryEditConfig, optimize_memory_ot


class FrozenVideoFlowPolicy(nn.Module):
    """Small differentiable proxy used to validate the Gate-0 causal protocol."""

    def __init__(self, feature_dim: int = 16, horizon: int = 8, action_dim: int = 2):
        super().__init__()
        self.horizon, self.action_dim = horizon, action_dim
        self.attention = nn.Linear(feature_dim, 1, bias=False)
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 64), nn.Tanh(), nn.Linear(64, horizon * action_dim)
        )
        generator = torch.Generator().manual_seed(17)
        for parameter in self.parameters():
            parameter.data.normal_(generator=generator, std=0.22)
            parameter.requires_grad_(False)

    def forward(self, memory: torch.Tensor, noises: torch.Tensor) -> torch.Tensor:
        # memory: [T,V,D]. Current observation/instruction are intentionally absent/edit-proof.
        slots = memory.mean(dim=1)
        weights = torch.softmax(self.attention(slots).squeeze(-1), dim=0)
        summary = (weights[:, None] * slots).sum(dim=0)
        mean = self.decoder(summary).reshape(self.horizon, self.action_dim)
        # Late-flow clean-action look-forward proxy.
        return mean[None] + 0.22 * noises


@dataclass
class Gate0Metrics:
    original: float
    positive: float
    negative: float
    norm_matched_random: float
    improvement_rate: float
    critical_event_rank: int


def run_gate0(seed: int, device: str = "cuda:0", samples: int = 8) -> Gate0Metrics:
    torch.manual_seed(seed)
    dev = torch.device(device)
    time, views, dim, horizon, action_dim = 16, 2, 16, 8, 2
    policy = FrozenVideoFlowPolicy(dim, horizon, action_dim).to(dev).eval()
    memory = 0.35 * torch.randn(time, views, dim, device=dev)
    critical_event = seed % (time - 2) + 1
    cue = torch.zeros(dim, device=dev)
    cue[seed % 2] = 3.0
    memory[critical_event] += cue
    # Stale/distractor event makes the memory-dependent task nontrivial.
    memory[(critical_event + 5) % time] -= 0.8 * cue
    target_sign = 1.0 if seed % 2 == 0 else -1.0
    target = torch.stack(
        [torch.linspace(0, target_sign, horizon, device=dev), torch.zeros(horizon, device=dev)], dim=-1
    )
    noises = torch.randn(samples, horizon, action_dim, device=dev)

    def policy_fn(mem: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return policy(mem, eps)

    actions = policy_fn(memory, noises)
    returns = -((actions - target) ** 2).mean(dim=(1, 2))
    result = optimize_memory_ot(
        memory,
        noises,
        returns,
        policy_fn,
        MemoryEditConfig(rank=8, trust_radius=0.01, temporal_topk=4),
    )
    random_delta = torch.randn_like(result.delta)
    random_delta *= result.delta.norm() / random_delta.norm().clamp_min(1e-8)

    def paired_return(mem: torch.Tensor) -> torch.Tensor:
        return -((policy_fn(mem, noises) - target) ** 2).mean(dim=(1, 2))

    base_r = paired_return(memory)
    plus_r = paired_return(result.memory_plus)
    minus_r = paired_return(memory - result.delta)
    random_r = paired_return(memory + random_delta)
    rank = int((result.temporal_saliency > result.temporal_saliency[critical_event]).sum().item() + 1)
    return Gate0Metrics(
        original=float(base_r.mean().cpu()),
        positive=float(plus_r.mean().cpu()),
        negative=float(minus_r.mean().cpu()),
        norm_matched_random=float(random_r.mean().cpu()),
        improvement_rate=float((plus_r > base_r).float().mean().cpu()),
        critical_event_rank=rank,
    )

