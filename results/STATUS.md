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

Not yet established:

- strict-pair conditional-support gain with a sufficiently large independent pair set;
- non-oracle readout recovery of that support;
- closed-loop task-success improvement;
- an advantage of OT over best-of-N or return-weighted centroids.

## Next execution

Follow `RUN_NEXT.md`:

1. audit and expand strict pairs;
2. E13-B2 strict pair-level replication;
3. E13-D non-oracle readout recovery;
4. E13-C2 replicated mechanism decomposition;
5. E12-S2 held-out multi-corruption recovery;
6. E14-A open-loop OT, followed by paired simulator evaluation.
