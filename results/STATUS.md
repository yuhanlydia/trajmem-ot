# TrajMem-OT status

## Latest research direction

The project now studies the **shared-memory false-consensus / memory-hypothesis coverage problem** in memory-augmented diffusion VLAs.

A conventional sampler fixes one history memory and changes only diffusion noise:

\[
A_n=F_\theta(M,\epsilon_n).
\]

The new experiments explicitly separate coherent memory views from conditional action noise:

\[
A_{b,n}=F_\theta(M^{(b)},\epsilon_n).
\]

The core operator is the exact memory-to-action JVP:

\[
Y=J_MD,
\]

followed by a regularized response-subspace pullback:

\[
c^*=\arg\min_c\|Yc-u\|^2+\lambda\|c\|^2,
\qquad \Delta M=Dc^*.
\]

Implemented experiment stages:

- **E11:** exact JAX JVP versus central action secants, speed, and response-rank diagnostics;
- **E12:** reward-free clean/degraded-memory action recovery with random, history, and hybrid bases;
- **E13:** matched-compute memory-view versus diffusion-noise branching and variance decomposition;
- **E14 (existing core, not yet connected to real outcomes):** return-tilted Sinkhorn OT followed by memory pullback.

The real GPU E11–E13 experiments remain to be run. Their scripts are present, but no released-checkpoint result is claimed from the CPU development environment.

## Evidence already established by the original experiments

### Controlled synthetic video-memory system

Eight seeds and eight particles produced:

- original return: `-0.553794`;
- positive memory edit: `-0.551756`;
- negative direction: `-0.555864`;
- norm-matched random: `-0.553955`;
- per-particle improvement rate: `96.875%`.

This is a controlled mechanism check, not RoboMME task performance.

### Released frozen MME-VLA plumbing

- official sample dataloader read `27,452` samples;
- released `perceptual-framesamp-modul` checkpoint restored on one RTX 3090;
- the model produced finite action chunks of shape `[20, 8]`;
- fixed-noise repeated inference was bit exact;
- clearing an applied history delta restored the original action exactly;
- a history-only memory edit changed the frozen model's action;
- deterministic simulator reconstruction plus identical prefix replay produced identical branch starts;
- different memory-conditioned actions produced distinct final physical-state fingerprints.

Therefore the causal chain

\[
M\rightarrow A\rightarrow S'
\]

is established. The beneficial chain

\[
M^+\rightarrow R_{env}\uparrow
\]

is not yet established.

## Original E7 — scalar black-box finite differences

- backbone: frozen `pi0.5 / MME-VLA perceptual-framesamp-modul`;
- 64 real training-demonstration states, 16 per selected task;
- memory: `static_image_emb [512, 2048]`;
- 16 random rank-one probe directions;
- 64 norm-matched random controls for every nonzero radius;
- best descriptive radius `0.1%`:
  - `P(Q+ > Q0) = 56.25%`;
  - `P(Q+ > Q-) = 60.94%`;
  - `P(Q+ > Qrandom) = 58.03%`.

The effect was weak and heterogeneous. This diagnoses the combination of sparse random directions and scalar finite-difference scores. It is not evidence about the exact action-response JVP added in E11.

## Original compact E8

- four tasks × two episodes × four conditions = 32 branches;
- `M`, fixed random `probe_plus`, `probe_minus`, and norm-matched random;
- 150-step cap;
- all conditions had 25% success because `VideoUnmask` succeeded and the other tasks did not;
- dense returns were zero.

The probes were random sensitivity controls, not return-conditioned edits. The experiment validated paired closed-loop infrastructure but did not establish a beneficial method effect.

## Next execution

Follow [`../RUN_NEXT.md`](../RUN_NEXT.md). The first required real-model result is E11 exact-JVP validity and response-rank measurement on the released checkpoint. E13 should use a coherent matched-history basis; random bases are controls only.
