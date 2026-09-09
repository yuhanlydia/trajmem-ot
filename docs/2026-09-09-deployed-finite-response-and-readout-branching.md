# Deployed Finite-Response and Memory-Readout Branching

## Problem being solved

The released MME-VLA uses BF16 in the perceptual-memory and transformer modulation path. An AD tangent is mathematically exact for the transformed computation, but the transformed BF16 primal differed from the ordinary deployed forward. Consequently, infinitesimal JVPs did not predict finite deployed memory edits.

The FP32 shadow recovered JVP–chord agreement, so the failure was numerical/control-surface mismatch rather than a broken memory connection.

## Design decision

Separate three roles:

### Scientific phenomenon

Use complete, coherent history hypotheses and ordinary forward inference. No gradient is required.

### Deployment operator

Use finite response columns from actual quantized endpoints:

\[
Y_{sec,k}=
\frac{F_{BF16}(M_k^+)-F_{BF16}(M_k^-)}{2\|\delta_k\|},
\qquad
\delta_k=\frac{M_k^+-M_k^-}{2}.
\]

### Smooth mechanism diagnostic

Use FP32 JVP only to study local rank and action-controllable directions:

\[
Y_{32}=J_{M,32}D.
\]

## Experiments

### E13-B: Oracle history transplant

Simulate a wrong shared memory and ask whether adding the correct history hypothesis improves coverage of the correct-memory action distribution under equal sampling compute.

### E13-C: Readout-mask branching

Keep memory values fixed and expose different contiguous valid history spans. This is the simplest deployment-safe approximation to multiple memory views.

### E12-S: Finite-response recovery

Use a clean-memory policy as a reward-free self-teacher, corrupt history, and solve a regularized inverse problem through `Y_sec`. Evaluate on fresh diffusion noises.

### E14: OT pullback

After E12-S succeeds, replace the clean-memory residual with return-tilted trajectory OT and evaluate paired closed-loop success.

## Minimal falsifiable predictions

1. If the shared-memory support gap exists, E13-B should improve calibrated target-mode coverage over noise-only sampling from the wrong memory.
2. If simple readout selection is sufficient, E13-C should recover part of the E13-B gain.
3. If deployed finite response is a useful control operator, E12-S should improve fresh-noise recovery and beat negative/random controls.
4. If OT adds distributional value beyond selection, E14 should beat best-of-N and return-weighted centroids under matched rollout compute.
