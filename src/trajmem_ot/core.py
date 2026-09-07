from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class MemoryEditConfig:
    beta: float = 5.0
    sinkhorn_epsilon: float | None = None
    sinkhorn_iters: int = 80
    damping: float = 1e-2
    cg_iters: int = 30
    rank: int = 8
    trust_radius: float = 0.01
    temporal_topk: int | None = None


@dataclass
class MemoryEditResult:
    memory_plus: Tensor
    delta: Tensor
    transport: Tensor
    coupling: Tensor
    source_weights: Tensor
    target_weights: Tensor
    temporal_saliency: Tensor
    residual_norm: float


def trajectory_cost(actions: Tensor, velocity_weight: float = 0.25) -> Tensor:
    """Pairwise trajectory cost for actions shaped [N, H, A]."""
    flat = actions.flatten(1)
    position = torch.cdist(flat, flat).square() / actions.shape[1]
    if actions.shape[1] < 2:
        return position
    velocity = actions[:, 1:] - actions[:, :-1]
    velocity = torch.cdist(velocity.flatten(1), velocity.flatten(1)).square()
    return position + velocity_weight * velocity / (actions.shape[1] - 1)


def return_tilted_weights(returns: Tensor, beta: float) -> Tensor:
    normalized = (returns - returns.mean()) / returns.std(unbiased=False).clamp_min(1e-6)
    return torch.softmax(beta * normalized, dim=0)


def sinkhorn(p: Tensor, q: Tensor, cost: Tensor, epsilon: float, iters: int) -> Tensor:
    """Stable log-domain entropic OT coupling with marginals p and q."""
    log_p, log_q = p.clamp_min(1e-12).log(), q.clamp_min(1e-12).log()
    kernel = -cost / max(epsilon, 1e-8)
    log_u = torch.zeros_like(p)
    log_v = torch.zeros_like(q)
    for _ in range(iters):
        log_u = log_p - torch.logsumexp(kernel + log_v[None, :], dim=1)
        log_v = log_q - torch.logsumexp(kernel + log_u[:, None], dim=0)
    coupling = torch.exp(kernel + log_u[:, None] + log_v[None, :])
    return coupling


def conjugate_gradient(
    operator: Callable[[Tensor], Tensor], rhs: Tensor, iterations: int, tolerance: float = 1e-8
) -> tuple[Tensor, float]:
    x = torch.zeros_like(rhs)
    residual = rhs - operator(x)
    direction = residual.clone()
    residual_sq = torch.dot(residual.flatten(), residual.flatten())
    for _ in range(iterations):
        applied = operator(direction)
        denom = torch.dot(direction.flatten(), applied.flatten()).clamp_min(1e-12)
        alpha = residual_sq / denom
        x = x + alpha * direction
        residual = residual - alpha * applied
        new_residual_sq = torch.dot(residual.flatten(), residual.flatten())
        if new_residual_sq.sqrt() <= tolerance:
            residual_sq = new_residual_sq
            break
        direction = residual + (new_residual_sq / residual_sq.clamp_min(1e-12)) * direction
        residual_sq = new_residual_sq
    return x, float(residual_sq.sqrt().detach().cpu())


def _project_rank(delta: Tensor, rank: int) -> Tensor:
    matrix = delta.reshape(delta.shape[0], -1)
    if rank <= 0 or rank >= min(matrix.shape):
        return delta
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    return ((u[:, :rank] * s[:rank]) @ vh[:rank]).reshape_as(delta)


def _project_temporal(delta: Tensor, topk: int | None) -> Tensor:
    if topk is None or topk >= delta.shape[0]:
        return delta
    scores = delta.flatten(1).norm(dim=1)
    keep = scores.topk(topk).indices
    mask = torch.zeros(delta.shape[0], dtype=torch.bool, device=delta.device)
    mask[keep] = True
    return delta * mask.reshape((-1,) + (1,) * (delta.ndim - 1))


def optimize_memory_ot(
    memory: Tensor,
    noises: Tensor,
    returns: Tensor,
    policy_fn: Callable[[Tensor, Tensor], Tensor],
    config: MemoryEditConfig = MemoryEditConfig(),
) -> MemoryEditResult:
    """Pull a return-tilted trajectory OT direction back into history memory.

    `memory` contains history only. `policy_fn(memory, noises)` must return [N,H,A].
    Model parameters, current observation, instruction and noise particles remain fixed.
    """
    base = memory.detach().requires_grad_(True)
    actions = policy_fn(base, noises)
    if actions.ndim != 3 or actions.shape[0] != returns.numel():
        raise ValueError("policy_fn must return [N,H,A] matching returns")

    source = torch.full_like(returns, 1.0 / returns.numel())
    target = return_tilted_weights(returns, config.beta)
    cost = trajectory_cost(actions.detach())
    positive_cost = cost[cost > 0]
    epsilon = config.sinkhorn_epsilon
    if epsilon is None:
        epsilon = 0.05 * float(positive_cost.median().cpu()) if positive_cost.numel() else 0.05
    coupling = sinkhorn(source, target, cost, epsilon, config.sinkhorn_iters)
    barycenter = coupling @ actions.detach().flatten(1) / source[:, None]
    transport = (barycenter - actions.detach().flatten(1)).reshape_as(actions)

    def forward(m: Tensor) -> Tensor:
        return policy_fn(m, noises).flatten()

    weighted_transport = (transport * source[:, None, None].sqrt()).flatten()

    def jvp(vector: Tensor) -> Tensor:
        return torch.autograd.functional.jvp(forward, base, vector, create_graph=False)[1]

    def vjp(vector: Tensor) -> Tensor:
        output = forward(base)
        return torch.autograd.grad(output, base, vector, retain_graph=True)[0]

    particle_scale = source[:, None, None].sqrt().expand_as(actions).flatten()
    rhs = vjp(weighted_transport * particle_scale)

    def normal_operator(vector: Tensor) -> Tensor:
        projected = jvp(vector) * particle_scale
        return vjp(projected) + config.damping * vector

    delta, residual = conjugate_gradient(normal_operator, rhs, config.cg_iters)
    delta = _project_rank(delta, config.rank)
    delta = _project_temporal(delta, config.temporal_topk)
    max_norm = config.trust_radius * base.detach().norm().clamp_min(1e-8)
    delta = delta * torch.clamp(max_norm / delta.norm().clamp_min(1e-8), max=1.0)
    saliency = delta.flatten(1).norm(dim=1)
    return MemoryEditResult(
        memory_plus=(base.detach() + delta.detach()),
        delta=delta.detach(),
        transport=transport.detach(),
        coupling=coupling.detach(),
        source_weights=source.detach(),
        target_weights=target.detach(),
        temporal_saliency=saliency.detach(),
        residual_norm=residual,
    )

