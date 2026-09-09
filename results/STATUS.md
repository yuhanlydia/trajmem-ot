# TrajMem-OT scientific status

## Current hypothesis

Diffusion sampling under a fixed history memory explores conditional motor stochasticity,

\[
A_n=F_\theta(M,\epsilon_n),
\]

but may not cover action support associated with another plausible interpretation of history. The proposed phenomenon is **shared-memory conditional-support failure**.

## Evidence already established

- The released MME-VLA checkpoint and official sample run on a 16GB RTX A4000.
- Fixed-noise inference is deterministic.
- History-only intervention changes frozen-policy actions and deterministic simulator futures, establishing \(M\rightarrow A\rightarrow S'\).
- E11-D validates the smooth FP32-shadow JVP connection across eight states and one/ten flow steps (`min robot-8D cosine 0.99878/0.99989`).
- The legacy E13-B all-pair oracle result reported mean native-history support gain `+0.640625`, median `+1.0`, with `11/16` positive directions.
- The legacy E12-S run reported positive fresh-noise corruption recovery on `8/8` states, median `+0.13243`.
- The legacy E13-C masks produced large open-loop action separation, including median between-view fraction `0.65513` at `(B,N)=(4,2)`.

## Why the previous aggregates are not final

- E13-B treated two directions from one pair as independent and several pairs have substantial current-context mismatch. It is an exploratory oracle result, not final pair-level evidence.
- E13-C used one small noise block and no native-support target. The reported fraction is action separation under strong masks, not calibrated belief uncertainty or correct-mode recovery.
- E12-S used one random control matched before BF16 quantization. Post-quantization applied norms and fresh-noise random/negative controls were not matched.

## Corrections implemented

1. `matched_variance_decomposition` separates memory main effect, noise main effect, and interaction under seed-matched grids.
2. E13-C2 supports repeated noise blocks and independent native-history support calibration.
3. Pair auditing now creates pre-outcome `strict`/`moderate` tiers.
4. E13-B2 averages transplant directions within pair and uses pair-level bootstrap/sign tests, tolerance sensitivity, headroom recovery, and normalized effect size.
5. E13-D tests contiguous readout masks on an incorrect donor history without inserting the correct memory into candidate branches.
6. E12-S2 matches negative and multiple random controls in actual post-BF16 applied norm, evaluates all controls on fresh noises, and reports finite-response extrapolation fidelity.
7. E14-A accepts return-labelled noise manifests and pulls return-tilted trajectory OT through the deployed BF16 finite-response operator.

## Current claim boundary

Supported:

- history conditioning can change action support;
- full-history oracle branching is a promising open-loop upper bound;
- history readout is a strong action-control surface;
- deployed finite responses can recover behavior after synthetic memory corruption.

## Corrected-protocol execution on 2026-09-09

- Pair audit: `2/8` strict and `4/8` moderate legacy pairs.
- E13-B2 strict smoke: two pairs, mean pair gain `+0.375`, 95% bootstrap interval `[0.0,0.75]`; one pair was positive and one tied.
- E13-D strict smoke: mean gain `+0.16875`, interval `[-0.0125,0.35]`; one positive and one negative pair.
- E13-C2: eight states and five noise blocks. `(4,2)` mean memory-main fraction was `0.37489`, while mean native-support coverage gain was `-0.25938` with interval `[-0.28125,-0.23750]`.
- E12-S2 random rows: 20 held-out corruptions and 16 applied-norm-matched controls per corruption. Mean ours-minus-random was `+0.17031`, interval `[+0.16531,+0.17474]`.
- E12-S2 contiguous rows: mean ours-minus-random was `+0.06538`, interval `[+0.03562,+0.09515]`.
- E14 simulator smoke: blocked before reset by unavailable Vulkan support (`vk::createInstanceUnique: ErrorIncompatibleDriver`) under both GPU and documented CPU-renderer environment settings.

## Expanded E13-B2 strict replication

The pre-outcome execution-start miner produced 22 episode-disjoint candidates and the history-aware audit retained all 22 as strict. The preregistered primary pair-level result was null:

- mean coverage gain `+0.07386`, 95% CI `[-0.04830,+0.21591]`;
- median gain `-0.0625`;
- 8 wins, 12 losses, 2 ties; exact sign-test `p=0.5034`.

Nearest-support distance improved on average and tolerance multiplier `2.0` was weakly positive, but these are secondary sensitivity results. Per the decision table, non-oracle router scaling stops here. The required next dataset consists of identical serialized simulator/current states paired with distinct histories.

Two independent diffusion-noise replications (`seed=17` and `seed=27`) did not change that conclusion. After averaging all three blocks within each pair, the mean gain was `+0.07765`, the pair-bootstrap interval was `[-0.03409,+0.21117]`, the median was `-0.04167`, and the exact sign test counted `7/15/0` wins/losses/ties (`p=0.1338`). Pair effects were highly correlated across blocks (`r=0.927--0.947`), so the uncertainty is dominated by stable pair heterogeneity rather than diffusion-noise sampling.

## Non-oracle readout result

The complete 22-pair E13-D run rejects the preregistered contiguous-mask generator as a recovery method:

- mean pair coverage gain `-0.20227`, 95% CI `[-0.26706,-0.12898]`;
- median gain `-0.275`;
- 4 wins and 18 losses; exact sign-test `p=0.00434`.

An explicitly post-hoc history-language subset of 13 pairs gave the same direction (`-0.17596`, interval `[-0.26538,-0.07885]`). Across the tested `2/4`-view and `0.25/0.50/0.75` keep-fraction grid, no configuration recovered native support on average. This is evidence against contiguous hard masks, not against learned readout hypotheses or true counterfactual histories.

## Closed-loop runtime boundary

E14 cannot run in this container because it was launched with `NVIDIA_DRIVER_CAPABILITIES=compute,utility`. The Vulkan loader and Mesa drivers were installed and the exact NVIDIA `595.84` userspace library was tested, but NVIDIA Vulkan still reports `VK_ERROR_INCOMPATIBLE_DRIVER`; CPU `llvmpipe` is enumerated but rejected by SAPIEN as an unsupported physical device. A fresh container must expose `graphics` (for example `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` or `all`) before simulator reset and paired branch execution are possible.

Not yet established:

- strict-pair conditional-support gain with a sufficiently large independent pair set;
- non-oracle readout recovery of that support;
- closed-loop task-success improvement;
- an advantage of OT over best-of-N or return-weighted centroids.

## Next execution

The remaining decisive experiment requires a Vulkan-capable container: serialize one simulator/current state, attach distinct valid histories, rerun E13-B2 on those true counterfactuals, and execute E14 actions from paired simulator branches. Do not scale the failed contiguous-mask family on the present offline sample.
