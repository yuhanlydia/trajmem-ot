# Counterfactual Memory Branching Validation Design

## Objective

Strengthen the current E13-B, E13-C, and E12-S evidence, then test whether return-conditioned finite-response memory edits improve paired closed-loop RoboMME outcomes. The four experiments must separate history effects from current-context differences, reuse matched diffusion randomness, freeze all method choices before held-out evaluation, and retain negative and random controls.

## Shared contracts

- Use the released `perceptual-framesamp-modul/79999` checkpoint without weight changes.
- Evaluate only the first eight deployed robot-action channels for primary metrics.
- Reuse identical noise seeds for every paired method comparison.
- Record checkpoint, data, pair/state, configuration, and seed provenance in every report.
- Select pairs and hyperparameters before observing their primary held-out outcomes.
- Keep open-loop action coverage separate from closed-loop success claims.
- Store lightweight JSON reports in Git; keep checkpoints, datasets, actions, and other large arrays ignored.

## E13-B2: strict counterfactual history pairs

### Construction

Create each counterfactual from one simulator branch state. Both conditions share the exact current observation, robot state, instruction, simulator state, policy weights, and diffusion noise. Only the historical bundle changes:

\[
o_t^a=o_t^b,\qquad s_t^a=s_t^b,\qquad l^a=l^b,
\]

\[
\mathcal H_t^a\ne\mathcal H_t^b.
\]

The preferred source is deterministic replay to a common branch state followed by two semantically distinct pre-branch histories. If the released sample cannot produce such a pair, the runner must stop with an explicit asset/provenance error rather than silently fall back to approximately matched episodes.

### Validation

Before action sampling, require exact equality of serialized current simulator state, current front and wrist images, current robot state, instruction tokens, and branch step. Require a nonzero difference in at least one valid historical bundle component. Save the audit independently of action outcomes.

### Comparison

For both history directions, compare equal compute:

\[
(1\text{ wrong memory},8\text{ noises})
\quad\text{versus}\quad
(2\text{ memory hypotheses},4\text{ noises each}).
\]

Calibrate the correct-mode tolerance from an independent native-history reference block. Report coverage gain, nearest-target distance improvement, paired memory effect under the same noise, and cluster-bootstrap confidence intervals over counterfactual pairs.

## E13-C2: oracle-target readout-mask coverage

### Target provenance

E13-B2 must export an independent native-history target set for each strict counterfactual state. E13-C2 consumes those targets without regenerating or changing them.

### Allocations

Compare matched total action-sampling compute:

\[
(B,N)\in\{(1,8),(2,4),(4,2)\}.
\]

For every state, repeat the allocation grid over multiple independently seeded noise blocks. Within a block, reuse the same base seed set across allocations wherever the allocation permits paired samples.

### Statistics

Primary outcomes are calibrated oracle-target coverage and nearest-target distance. Fit a two-way crossed model with memory view and noise block as factors, including their interaction. Report state-cluster bootstrap confidence intervals for each allocation difference relative to `(1,8)`. Variance decomposition remains diagnostic and cannot substitute for target coverage.

## E12-S2: frozen finite-response recovery

### Frozen configuration

Use the state-0-selected configuration without retuning:

- basis: `history`;
- probe radius: `2.5e-4`;
- corruption: drop `20%` of valid history rows;
- maximum applied update radius: `2.5e-3`;
- existing ridge damping and basis cardinality.

### Evaluation grid

Use held-out states `4,5,6,7`, five predeclared corruption seeds per state, and four fresh diffusion noises per corruption. Generate at least 16 independent norm-matched random edits for every state/corruption case. Apply ours, negative, and random edits to the same corrupted memory and reuse noise seeds.

### Statistics

The primary statistic is the paired difference:

\[
\Delta_{paired}=\Delta_{ours}-\Delta_{random}.
\]

Bootstrap by state and corruption cluster, preserving all fresh-noise observations within each cluster. Report the 95% interval, win fraction, per-state results, negative-direction recovery, applied BF16 edit norm, changed-coordinate fraction, and linear-prediction residual. A positive median alone is insufficient; the primary success condition is a strictly positive lower confidence bound for `ours - random`.

## E14: return-conditioned closed-loop evaluation

### Trajectory collection

At each frozen simulator branch state, sample matched-compute action trajectories from the original memory and record the full action chunk, environment return, success, subgoal progress, initial simulator-state fingerprint, memory provenance, and diffusion seed. Trajectories used to form the target must be disjoint from evaluation rollouts.

### Action targets

Construct and retain three candidate targets:

1. best-of-`N` trajectory;
2. return-weighted action centroid;
3. entropic trajectory-OT barycentric transport.

For OT, normalize returns only within the same branch state and use a fixed predeclared `beta`, cost scaling, entropy regularization, and Sinkhorn tolerance. The runner must reject missing, nonfinite, or constant return vectors.

### Pullback

Build the deployed BF16 finite-response matrix at the branch state, then solve the frozen ridge pullback for centroid and OT targets. Quantize and measure the actual applied edit before closed-loop replay. Never use the unvalidated deployed BF16 AD tangent.

### Conditions

Run matched-seed paired rollouts for:

- frozen original;
- best-of-`N` selection;
- return-weighted centroid selection;
- finite-response plus centroid;
- finite-response plus OT;
- norm-matched random edit;
- negative finite-response direction.

All conditions begin from the same serialized simulator state. Compare equal rollout-compute budgets and separately disclose the offline target-construction cost.

### Endpoints

Primary endpoints are paired simulator success and subgoal progress. Secondary endpoints are return, time to subgoal, edit magnitude, and action-target realization error. Report state-cluster bootstrap confidence intervals and paired sign tests. E14 supports the method only if finite-response plus OT improves held-out paired success or subgoal progress over frozen original and matched-compute selection baselines; return-only gains are diagnostic.

## Execution order and stopping rules

1. Build and audit strict E13-B2 pairs. Stop B2 if exact current-context equality cannot be established.
2. Run B2 smoke, then the full predeclared pair set.
3. Export independent oracle targets and run C2 with multiple noise blocks.
4. Run S2 on the frozen held-out grid and compute the paired bootstrap interval.
5. Implement E14 collection and replay contracts, run deterministic smoke, then run held-out paired closed-loop evaluation.

Any checkpoint mismatch, data mismatch, simulator fingerprint mismatch, nonfinite probability/action/return, unequal paired seed set, or incomplete result file is a hard failure. Runners must resume only at whole atomic experimental units and must never mix partial output into an aggregate.

## Claim boundaries

- B2 can establish an open-loop conditional-support effect under strict counterfactual histories.
- C2 can establish whether fixed mask views recover oracle action support.
- S2 can establish whether deployed finite-response recovery beats norm-matched random edits across corruptions.
- Only E14 can establish improved paired closed-loop success or subgoal progress.

