# Native interference integration plan

Execute inline; this continues the authorized FRMD experiment implementation.
Research continuation has no fixed metric gate. Do not change the frozen policy.

1. Finish verified numeric-first selection of 10 source episodes per task.
   Match each raw setup seed/difficulty to benchmark metadata. These released raw
   episodes are training-split data: results must be labeled development/pilot,
   never held-out generalization. Preserve the split and revision in provenance.
2. Preprocess all 30 episodes in one new output directory with the pinned upstream
   SigLip encoder. Record the actual file iteration order and per-episode index
   ranges. Never overwrite the existing official sample or fabricated raw indices.
3. Build a catalog of actual raw demo prefixes, query steps, raw source paths,
   preprocessed episode IDs, and first-execution query indices. Use fixed numeric
   selection, independent of policy outputs. Include clean k=0 and distractor
   levels 3/7; choose unrelated task demos deterministically from other episodes.
4. Concatenate the relevant demo, k distractor demos, and recent execution frames.
   Recompute native token-dropping scores from raw pixels across this sequence,
   including cross-session boundaries. Reuse cached frozen visual features,
   regenerate chronological positional embeddings, and preserve token-source IDs.
   The teacher uses relevant demo plus recent frames only. Current image, state,
   instruction, weights, and diffusion noise remain paired.
5. Extend a separate native runner around the tested repair core. Save all raw
   indices and hashes. Run paired two-step/one-step, centroid, OT, random, negative,
   interpolation, and retrieved teacher. Preserve all outcomes and query counts.
6. Evaluate persistence at later observations. No teacher is used to choose new
   edits; teacher calls are offline scoring only. Transfer applied deltas using
   token-source identity across buffer updates and report retained edit norm and
   token fraction. Report if the buffer evicts the repaired tokens.
7. Map source setup seeds to renderer-free train environments and replay recorded
   execution prefixes. Verify start fingerprints, unnormalize model actions via
   the policy output transform, and compare branch success/progress/return.
   Keep graphics-dependent full closed loop separate until rendering works.
8. Analyze episode/task aggregates and reuse across settings. No numerical outcome
   causes automatic program termination; weak/negative results trigger analysis
   and debugging without being relabeled positive or excluded.

## Frame-sampling reuse and closed-loop integration

The same released VLA family now has an independently verified frame-sampling
checkpoint and encoder. Preserve method/configuration separation and reuse the
repair core unchanged. Continue all planned ablations regardless of metric gates.

1. Verify frame-sampling output against upstream at short padded and long sampled
   histories on all three tasks; distinguish pooled token resolutions in identity.
2. Use a process-local NVIDIA EGL Vulkan ICD and CPU tensor readback for the
   benchmark environment. Keep visual geometry and real RGB observations enabled.
3. Run the policy and simulator in their respective pinned Python environments,
   exchanging local NPZ observations/actions through a persistent worker. Reuse
   source history features; encode every new executed frame with the frozen policy
   vision encoder. Apply the original memory delta only to surviving source tokens
   after each update. Teacher history is available only to the teacher condition.
4. Compare six branches from identical physical fingerprints with matched noise
   schedules. Record current observations, action digests, applied-edit retention,
   subgoal progress, terminal status and query budgets. Offline repair seeds and
   new closed-loop evaluation seeds remain distinct. This first closed-loop phase
   is a bounded multi-query development rollout, not a full benchmark claim.
