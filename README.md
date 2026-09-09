# TrajMem-OT

**Memory-hypothesis branching and trajectory-to-memory transport for memory-augmented diffusion VLAs.**

## Research question

A memory-augmented diffusion VLA usually encodes one history and spends test-time compute by changing only diffusion noise:

\[
M_t=E(H_t),\qquad A_n=F_\theta(M_t,\epsilon_n).
\]

Those samples explore **how to act under one shared interpretation of the past**. They do not necessarily cover uncertainty over **which interpretation of the past is correct**. TrajMem-OT separates these axes:

\[
A_{b,n}=F_\theta(M_t^{(b)},\epsilon_n),
\]

where `b` indexes a coherent memory/readout hypothesis and `n` indexes conditional action noise.

The central question is:

> Can branching over coherent history interpretations recover action support that additional diffusion samples under one shared memory cannot reach?

## What is established

On the released `perceptual-framesamp-modul/79999` checkpoint and RoboMME sample:

- history-only interventions causally change frozen-policy actions and simulator futures;
- FP32-shadow JVP diagnostics match finite chords over eight states at one and ten flow steps (`min robot-8D cosine = 0.99878/0.99989`);
- the original E13-B oracle transplant produced an exploratory all-pair mean native-support gain of `+0.640625` over 16 directions;
- contiguous readout masks strongly separate action distributions (`legacy median between-view fraction = 0.65513` at `(B,N)=(4,2)`);
- deployed BF16 finite-response pullback produced positive fresh-noise corruption recovery on `8/8` states (`median +0.13243`).

These results are promising, but the previous protocols had three important limitations:

1. E13-B pooled two directions from each pair as if they were independent and included several poorly matched current contexts.
2. E13-C reported a single small-noise-block separation statistic, not calibrated recovery of native-history support.
3. E12-S used one random control whose norm was matched before BF16 quantization, not after the actual deployed cast.

The current code corrects these limitations. Existing JSON files remain as historical evidence and must not be reinterpreted as results from the corrected protocols.

## Numerical resolution

The released BF16 memory-modulation/LLM path is not faithfully described by an infinitesimal AD tangent. An FP32 shadow validates the JVP wiring on a smooth diagnostic path, but deployment uses finite BF16 interventions. The repository therefore separates three operators:

- **FP32 JVP:** mechanism/rank diagnostic only;
- **BF16 finite response:** deployed inverse-control operator;
- **full history transplant/readout masks:** gradient-free tests of the support gap.

For structured direction `d_k`, the deployed response is

\[
M_k^\pm=Q_{BF16}(M\pm h d_k),\qquad
\delta_k=\frac{M_k^+-M_k^-}{2},
\]

\[
r_k=\frac{F_{BF16}(M_k^+)-F_{BF16}(M_k^-)}{2},\qquad
Y_{sec}=[r_k/\|\delta_k\|]_k.
\]

Given an action-space target `u`, the memory update is solved as

\[
c^*=\arg\min_c\|Y_{sec}c-u\|_2^2+\lambda\|c\|_2^2,
\qquad
\Delta M=\sum_k c_k\frac{\delta_k}{\|\delta_k\|}.
\]

SVD is applied to the **action-response matrix** `Y_sec`, not repeatedly to the full raw memory tensor.

## Corrected experiments

### E13-B2 — audited oracle history transplant

Pairs are audited before outcomes using named `strict` and `moderate` current-context tiers. The two transplant directions are averaged within each pair; bootstrap intervals and sign tests resample pairs. Results are reported as **native-history-conditioned support**, not ground-truth task correctness. A predeclared tolerance-multiplier curve and headroom-normalized recovery are included.

### E13-C2 — matched readout branching

All memory values remain unchanged; only valid history spans are exposed through boolean masks. The same diffusion seeds are reused across views. A balanced two-way decomposition reports:

\[
V_{total}=V_{memory}+V_{noise}+V_{interaction}.
\]

Independent noise blocks estimate stability. Optional native-history reference/target sets measure support coverage rather than variance alone.

### E13-D — non-oracle readout recovery

This is the decisive bridge from oracle to automatic views. Starting from an incorrect donor history, contiguous readout masks try to recover the native-history-conditioned action support **without inserting the correct memory into the candidate set**. Several keep fractions are compared under equal trajectory compute.

### E12-S2 — quantization-aware finite-response recovery

Corruption seeds and modes are explicit. Negative and multiple random controls are matched to the method edit in **actual post-BF16 applied norm**. The same controls are evaluated under fresh diffusion noise. The report includes actual-versus-linear response fidelity and state-level paired inference.

### E14-A — manifest-driven trajectory OT pullback

A rollout manifest supplies noise seeds and scalar returns. Return-tilted OT constructs a multi-particle action transport, which is pulled back through `Y_sec`. This is an open-loop target-fit experiment; paired simulator execution is still required for any return/success claim.

## Key files

```text
configs/e13_pair_quality.yaml          pre-outcome strict/moderate pair tiers
configs/e14_manifest.example.json      return-labelled OT manifest contract
src/trajmem_ot/finite_response.py      BF16 finite response + quantized controls
src/trajmem_ot/hypothesis_metrics.py   calibrated support and tolerance curves
src/trajmem_ot/memory_views.py         matched memory/noise decomposition
src/trajmem_ot/pair_quality.py         context-quality assessment
src/trajmem_ot/stats.py                pair/state-level bootstrap and sign tests
src/trajmem_ot/transport.py            NumPy return-tilted trajectory OT
scripts/audit_e13_pairs.py             pre-outcome pair audit and approved files
scripts/run_e13_oracle_transplant.py   E13-B2
scripts/run_e13_readout_mask_branching.py  E13-C2
scripts/run_e13_pair_readout_recovery.py   E13-D
scripts/run_e12_secant_recovery.py     E12-S2
scripts/run_e14_ot_pullback.py         E14-A
RUN_NEXT.md                            exact 16GB execution handoff
results/STATUS.md                      claim ledger
```

## Verification

```bash
python -m pip install -e '.[dev,jax]'
pytest -q
python -m compileall -q src scripts tests
```

Released-checkpoint experiments require the upstream RoboMME/OpenPI runtime, checkpoint, and preprocessed sample. Follow [`RUN_NEXT.md`](RUN_NEXT.md).

## Claim boundary

The repository does **not** yet establish improved RoboMME task success, a calibrated posterior over memory hypotheses, or an OT advantage over best-of-N in closed loop. The next decisive evidence is: strict-pair E13-B2, non-oracle E13-D support recovery, multi-corruption E12-S2, and paired simulator evaluation of E14 edits.
