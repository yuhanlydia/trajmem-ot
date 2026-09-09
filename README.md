# TrajMem-OT

**Memory-hypothesis branching and trajectory-to-memory transport for memory-augmented diffusion VLAs.**

## Research question

A memory-augmented diffusion VLA normally compresses one interaction history into one internal memory and then spends test-time compute by varying only action noise:

\[
M_t=E(H_t),\qquad A_n=F_\theta(M_t,\epsilon_n).
\]

Those samples answer **how to act under one shared interpretation of the past**. They do not necessarily cover uncertainty over **which interpretation of the past is correct**. If the shared memory is wrong or ambiguous, many trajectories can agree while inheriting the same error.

TrajMem-OT separates the two axes:

\[
A_{b,n}=F_\theta(M_t^{(b)},\epsilon_n),
\]

where `b` indexes a coherent history-memory hypothesis or readout view and `n` indexes conditional diffusion noise.

The central scientific question is therefore:

> Can branching over coherent history interpretations recover action modes that cannot be reached by drawing more diffusion samples from one shared memory?

## What the latest diagnostics changed

The released `perceptual-framesamp-modul/79999` checkpoint stores and processes the history path largely in BF16. On that deployed path, an infinitesimal AD tangent did not predict finite memory interventions. The discrepancy was deterministic, survived quantization-aware endpoints, persisted in the real eight robot channels, and appeared at the memory-modulation/LLM boundary.

An FP32 shadow diagnostic then promoted the perceptual memory encoder and the LLM embedding/accumulation path while keeping the checkpoint weights fixed. On state 0 with one flow step it reduced the transformed-primal discrepancy to `3.55e-6` and produced:

- robot-channel JVP–chord cosine: `0.99970`;
- all-channel JVP–chord cosine: `0.99924`;
- relative error: `0.0393`.

This strongly supports the JVP wiring and mathematical connection. It also shows that **raw BF16 memory is not a reliable infinitesimal deployment control variable**.

The repository now uses a split operator policy:

- **FP32 JVP** is a smooth diagnostic/oracle for mechanism analysis;
- **BF16 finite secants** are the authoritative operator for deployed memory edits;
- **history-bundle transplant and readout-mask branching** test the scientific phenomenon without requiring any gradient.

## Current method stack

### 1. Oracle full-history transplant

For a matched pair of histories, keep the current image, current robot state, instruction, model, and noise fixed. Replace the entire historical bundle:

\[
\mathcal H_M=
\{M^{img},M^{pos},M^{state},m^{mask}\}.
\]

This gives a clean upper-bound experiment:

\[
F(o_t,l,\mathcal H_M^{wrong},\epsilon)
\quad\text{versus}\quad
F(o_t,l,\mathcal H_M^{correct},\epsilon).
\]

E13-B compares equal compute:

\[
(1\text{ wrong memory},N\text{ noises})
\quad\text{versus}\quad
(2\text{ memory hypotheses},N/2\text{ noises each}).
\]

The target tolerance is calibrated from two independent samples of the correct-memory policy. The main output is correct-mode coverage gain, not raw action variance.

### 2. Deployment-faithful readout branching

Instead of modifying one million BF16 memory values, E13-C keeps all memory values fixed and changes only which valid history span may be read:

\[
m_t^{(b)}\subseteq m_t.
\]

The initial implementation uses contiguous boolean history-mask views. It is a parameter-free baseline for the eventual low-dimensional FP32 attention-bias router:

\[
\alpha^{(b)}=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt d}+Uz_b
\right).
\]

### 3. Deployed finite-response pullback

For each structured memory direction `d_k`, form the actual quantized endpoints and record the deployed action half-chord:

\[
M_k^\pm=Q_{BF16}(M\pm h d_k),
\]

\[
\delta_k=\frac{M_k^+-M_k^-}{2},
\qquad
r_k=\frac{F_{BF16}(M_k^+)-F_{BF16}(M_k^-)}{2}.
\]

After normalizing by \(\|\delta_k\|\), the response matrix is

\[
Y_{sec}=[r_1/\|\delta_1\|,\ldots,r_K/\|\delta_K\|].
\]

Given a desired robot-action change `u`, solve

\[
c^*=\arg\min_c
\|Y_{sec}c-u\|_2^2+\lambda\|c\|_2^2,
\]

\[
\Delta M=\sum_k c_k\frac{\delta_k}{\|\delta_k\|}.
\]

The resulting edit is clipped, quantized, and evaluated through the real BF16 policy. SVD is applied to the **finite action-response matrix**, never repeatedly to the full raw memory tensor.

### 4. Trajectory OT

Once the phenomenon and deployed response operator are validated, multi-future returns can define a target trajectory distribution:

\[
q_i\propto\exp(\beta R_i),
\qquad
\Gamma^*=\operatorname{Sinkhorn}(p,q,C).
\]

The barycentric action transport `u_OT` is then pulled back through `Y_sec` or a validated FP32 readout operator. Environment reward need not be differentiable.

## Evidence so far

### Established

- The released MME-VLA checkpoint and official 80-episode sample run on a 16GB RTX A4000.
- History-only intervention changes the frozen action chunk.
- Those action changes create different simulator futures under deterministic branch replay.
- The causal chain \(M\rightarrow A\rightarrow S'\) is established.
- Exact JAX JVP execution is feasible on the released checkpoint.
- FP32 shadow analysis localizes the earlier JVP mismatch to low-precision memory-encoder/LLM arithmetic and validates the smooth JVP connection on the diagnostic path.
- E11-D replicated the FP32-shadow result over eight states at one and ten flow steps. Robot-8D JVP--chord cosine was at least `0.99878` for one step and `0.99989` for ten steps.
- E13-B completed eight pre-audited history pairs in both directions. Across 16 directions, oracle history branching achieved mean correct-mode coverage gain `+0.6406`, median gain `+1.0`, with `68.75%` positive directions under matched trajectory compute.
- E13-C completed eight states. At allocation `(4 memory views, 2 noises/view)`, contiguous readout masks produced median robot-8D memory-variance fraction `0.6551`.
- E12-S froze the history basis and probe radius after state 0, then evaluated all eight states. Fresh-noise recovery was positive on `8/8` states, with median `+0.1324`, versus median random-control recovery `-0.0030` and negative-direction recovery `-0.1516`.

### Preliminary or negative

- The old 16-direction scalar finite-difference estimator was weak and heterogeneous.
- The sampled response energy appeared concentrated (`median r90=2` on eight BF16 runs; `r90=3` in the first FP32 shadow), but this is not a claim about the full Jacobian rank.
- E13-A used tiny global history-basis edits and found that diffusion noise dominated action variance. This does not test complete competing history hypotheses.

### Not established

- environment success improvement;
- a learned FP32 readout router;
- trajectory-OT improvement on the released model.

The E13-B result establishes an open-loop conditional-support gap for the audited sample pairs. It does not by itself establish improved closed-loop task success. E13-C currently measures action-distribution separation because no calibrated target-action file was supplied; it does not yet show that contiguous masks recover the correct oracle mode.

## Repository layout

```text
configs/                              16GB/24GB presets and pair examples
src/trajmem_ot/finite_response.py     deployed BF16 secant-response operator
src/trajmem_ot/hypothesis_metrics.py  calibrated set-coverage metrics
src/trajmem_ot/memory_views.py        latent views and history-mask views
src/trajmem_ot/robomme_jax.py         safe history-only interventions
src/trajmem_ot/robomme_runtime.py     one-checkpoint multi-state loader
scripts/run_e11_robomme_jvp.py        BF16 and FP32-shadow diagnostics
scripts/run_e11_stage_localization.py layer/stage localization
scripts/run_e12_secant_recovery.py    reward-free deployed secant recovery
scripts/analyze_e12_secant.py         aggregate deployed recovery sweeps
scripts/audit_e13_pairs.py            pre-outcome pair/context audit
scripts/run_e13_oracle_transplant.py  oracle full-history hypothesis test
scripts/run_e13_readout_mask_branching.py
                                      history-readout branching baseline
scripts/analyze_e13_oracle.py         aggregate oracle coverage gains
scripts/analyze_e13_readout.py        aggregate readout-view diagnostics
RUN_NEXT.md                           exact 16GB/24GB execution handoff
results/STATUS.md                     claim ledger and current evidence
```

## Verification

```bash
python -m pip install -e '.[dev,jax]'
pytest -q
python -m compileall -q src scripts tests
```

The real checkpoint experiments require the upstream RoboMME/OpenPI runtime, released checkpoint, and preprocessed sample data. Follow [`RUN_NEXT.md`](RUN_NEXT.md).

## Compute policy

The 16GB path runs one environment state at a time. The deployed secant experiment uses eight structured directions sequentially; the oracle transplant uses eight candidate trajectories per condition; the readout experiment uses matched allocations `(B,N)=(1,8),(2,4),(4,2)`. The `(8,1)` allocation is not used as evidence for variance decomposition because within-view noise cannot be estimated from one sample.
