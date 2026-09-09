# TrajMem-OT scientific status

## Current gap hypothesis

Diffusion sampling under one fixed history memory explores conditional action stochasticity:

\[
A_n=F_\theta(M,\epsilon_n).
\]

It may fail to cover futures associated with a different plausible interpretation of history. The proposed phenomenon is **shared-memory conditional-support failure**, not generic action variance.

## Evidence established

- Released MME-VLA checkpoint and sample data run on a 16GB RTX A4000.
- Fixed-noise repeated inference is deterministic.
- A history-only intervention changes the frozen policy action.
- Different memory-conditioned actions create different deterministic simulator futures.
- Therefore \(M\rightarrow A\rightarrow S'\) is established.
- E11-D replicated FP32-shadow JVP validation on eight released-checkpoint states at one and ten flow steps. The minimum robot-8D cosine was `0.99878` and `0.99989`, respectively.
- E13-B ran eight pairs in both directions. Mean calibrated correct-mode coverage gain was `+0.640625`, median gain was `+1.0`, and `11/16` directions were positive.
- E13-C ran eight states. For `(B,N)=(4,2)`, the median robot-action memory-variance fraction was `0.65513`; this is a separation result because no target-action set was supplied.
- E12-S selected the history basis and `2.5e-4` probe radius on state 0, froze the configuration, and evaluated states 1--7. Fresh-noise recovery was positive for all eight states with median `+0.13243`; the random and negative controls had medians `-0.00304` and `-0.15160`.

## Raw BF16 JVP diagnosis

Eight-state E11 execution succeeded and the sampled response energy was concentrated (`median effective_rank_90 = 2`). However, both nominal and quantization-aware BF16 chords disagreed with the AD tangent.

E11-C localized the transformed-primal mismatch to the memory-modulation/LLM path. The memory encoder alone had zero transformed-primal difference, while the first modulation velocity showed nonzero primal drift.

The FP32 shadow promoted the perceptual memory encoder and LLM embedding/accumulation path and used highest matmul precision. On state 0 with one flow step:

- transformed-primal difference: `3.55e-6`;
- robot-8D JVP–chord cosine: `0.99970`;
- all-channel cosine: `0.99924`;
- relative error: `0.0393`.

Interpretation: the JVP wiring is strongly supported on a smooth diagnostic path, but the released BF16 path must not be treated as an infinitesimal deployment control surface.

## E13-A result

Tiny global history-basis edits at relative radius `2.5e-4` produced very small robot-action memory variance fractions on two states (`0.00019–0.00171`). Diffusion noise dominated. This is a null result for small generic perturbations, not a test of complete competing history hypotheses.

## Resolution implemented in the current code

1. **FP32 JVP remains diagnostic only.**
2. **Full history-bundle transplant** tests the gap without gradients (`E13-B`).
3. **History-mask branching** changes readout support without changing BF16 memory values (`E13-C`).
4. **Finite BF16 secant responses** replace AD-JVP for deployed inverse pullback (`E12-S`).
5. SVD is performed on action-response matrices, not on all raw memory values at every step.

## Open claims

Not yet established:

- oracle memory branching improves closed-loop task success;
- contiguous masks recover calibrated correct modes from the oracle experiment;
- an automatic learned readout view generator recovers the oracle gain;
- trajectory OT improves simulator return or success;
- memory branching outperforms matched-compute diffusion-only sampling in closed loop.

E14 cannot yet be executed from this checkout: the repository contains the OT core but no released-checkpoint E14 runner or return-labelled paired simulator trajectories. Those are required to turn the validated E12-S control operator into a return-conditioned experiment.

## Next execution

Follow `RUN_NEXT.md` in this order:

1. E11-D FP32-shadow replication;
2. E13-B oracle history transplant;
3. E13-C readout-mask branching;
4. E12-S deployed secant recovery;
5. E14 return/OT only after a useful control operator exists.
