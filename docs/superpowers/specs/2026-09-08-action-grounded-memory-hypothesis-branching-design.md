# Action-Grounded Memory-Hypothesis Branching Design

## Research objective

Memory-augmented diffusion VLAs normally sample many action trajectories while holding one encoded history memory fixed. This explores conditional motor stochasticity but not uncertainty about the history interpretation itself. The project tests and repairs the **shared-memory false-consensus** failure: many rollouts may agree because they inherit the same wrong memory hypothesis.

The implementation must preserve the existing TrajMem-OT results and add a white-box operator path for the released RoboMME pi0.5/MME-VLA checkpoint.

## Scientific decomposition

For fixed current observation, instruction, robot state, policy parameters, and diffusion noise, define

\[
A = F_\theta(M,\epsilon), \qquad J_M=\partial A/\partial M.
\]

The system separates three questions:

1. **Memory basis:** which coherent directions represent alternative readings of history?
2. **Action grounding:** which directions have functional authority over the action chunk, measured by exact JAX JVPs `J_M d`?
3. **Behavior target:** where should the action distribution move, supplied initially by demonstration residuals and later by return-weighted or OT targets?

The first runnable method is JVP-SVD pullback. Given a structured memory basis `D=[d_1,...,d_K]`, compute

\[
Y=J_MD,
\]

then solve

\[
c^*=\arg\min_c\|Yc-u\|_2^2+\lambda\|c\|_2^2,
\qquad \Delta M=Dc^*.
\]

A truncated SVD of `Y`, not of the raw memory, identifies the action-controllable combinations of memory directions.

## Experiment sequence

### E11 — exact JVP correctness and throughput

Run the released perceptual `framesamp-modul` checkpoint in-process. Compare exact JAX JVPs against central action secants at fixed noise. Report cosine agreement, relative error, first-call compile time, warmed latency, and peak device memory.

### E12 — operator and basis comparison

Compare scalar-Q finite difference, full-action finite-difference SVD, and exact JVP-SVD. Compare random rank-one and history-difference bases. Use demonstration action residual only as an operator diagnostic. Evaluate held-out states and fresh diffusion-noise seeds.

### E13 — memory-view versus diffusion-noise branching

At matched total sampling budget `B*N`, compare `(B,N)=(1,32),(4,8),(8,4),(32,1)` where `B` is the number of coherent memory views and `N` is the number of action-noise samples per view. Report within-memory action variance, between-memory action variance, target-mode coverage, and false-consensus statistics.

### E14 — return-tilted OT pullback

After E11–E13 validate the operator and phenomenon, use existing return-tilted Sinkhorn transport to form an action-space target and pull it back through the JVP response subspace. Do not conflate operator quality with return-target quality.

## Interfaces

- `trajmem_ot.jax_operator`: exact/chunked JVPs, central secants, response SVD, ridge pullback.
- `trajmem_ot.memory_basis`: random rank-one and history-difference bases.
- `trajmem_ot.memory_views`: view construction and law-of-total-variance diagnostics.
- `trajmem_ot.presets`: explicit 16GB and 24GB settings.
- `trajmem_ot.robomme_jax`: optional in-process adapter for upstream RoboMME/MME-VLA objects.

The package must import without RoboMME or JAX-specific upstream packages installed. Upstream imports occur only inside runtime adapter functions.

## Hardware constraints

### 16GB preset

- one environment state at a time;
- 8 basis directions;
- JVP direction chunks of 2;
- 4 memory views and 2 noise samples per view;
- response rank 4;
- BF16 model forward where supported, FP32 response/SVD accumulation;
- XLA client memory fraction 0.75.

### 24GB preset

- one environment state at a time;
- 16 basis directions;
- JVP direction chunks of 4;
- 8 memory views and 4 noise samples per view;
- response rank 8;
- XLA client memory fraction 0.88.

Batching prioritizes JVP directions and noise particles within one state, because model parameters and current inputs are shared.

## Causal scope

Only the historical memory field may change. Current images, current robot state, instruction tokens, model parameters, and specified diffusion noise must remain fixed. Every real-model result records content fingerprints and noise identifiers.

## Safety and interpretation

- SVD is a numerical solver and action-controllability diagnostic, not the scientific contribution by itself.
- Demonstration residual experiments diagnose the memory-to-action operator; they are not environment-return claims.
- Random memory noise is a control, not a coherent hypothesis generator.
- Memory-view branching should use actual paired histories or history-derived bases before learned phrase-gradient views.
- Old E7/E8 results remain archived and are not silently relabeled as results of the new method.
