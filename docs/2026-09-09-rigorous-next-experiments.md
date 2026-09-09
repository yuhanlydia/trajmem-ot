# Rigorous next experiments after the first positive TrajMem-OT results

## Why this revision is necessary

The first positive results established useful signals, but three statistical/finite-precision confounds remained: pair directions were pooled as independent samples, readout variance used too few noise replications to support a strong uncertainty interpretation, and random memory controls were not matched after BF16 quantization.

The revised design keeps the scientific question unchanged while making each inference unit explicit.

## E13-B2: pair-level support diagnosis

The experimental unit is a matched history pair, not a transplant direction. Two directions are averaged within pair. Pair eligibility is determined before outcomes by current image/state similarity, prompt compatibility, mask compatibility, and a nontrivial history contrast. The output is native-history-conditioned support coverage, not ground-truth correctness.

Primary estimand:

\[
\Delta_{pair}=\frac{1}{2}\sum_{d\in\{a\leftarrow b,b\leftarrow a\}}
\left(C^{branch}_d-C^{noise}_d\right).
\]

Bootstrap and sign tests resample pairs. Sensitivity is reported over a fixed tolerance-multiplier grid.

## E13-C2: balanced two-way decomposition

For the seed-matched grid \(A_{b,n}\), report

\[
A_{b,n}=\mu+\alpha_b+\beta_n+\gamma_{b,n},
\]

with

\[
V_{total}=V_\alpha+V_\beta+V_\gamma.
\]

Independent noise blocks quantify stability. The memory-main fraction is an action-separation statistic only.

## E13-D: automatic readout upper baseline

Full correct memory is no longer inserted into candidate branches. Starting from a donor history, contiguous readout windows create parameter-free views. The target set comes from independent native-history samples. This asks whether a simple deployment-safe view generator recovers any of the oracle support gap.

## E12-S2: finite-precision causal controls

Every negative/random control is searched until its post-cast BF16 edit norm matches the method edit. Controls are carried unchanged across fresh diffusion seeds. Corruption seeds are averaged within state before state-level inference.

The finite response is also tested for extrapolation:

\[
\operatorname{cos}(Y_{sec}c, F(M+\Delta M)-F(M)),
\]

and

\[
\frac{\|F(M+\Delta M)-F(M)-Y_{sec}c\|}
{\|F(M+\Delta M)-F(M)\|}.
\]

Poor fidelity implies the update leaves the locally measured secant regime and should be reduced or iteratively relinearized.

## E14-A/B: OT target then causal return

E14-A only tests whether a return-tilted transport field can be realized by the finite-response control surface. E14-B must execute saved action branches from identical simulator states. Only E14-B can support a return or success claim.
