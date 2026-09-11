# Tokendrop ablation progress

Snapshot: 2026-09-11T15:28:31.912186+00:00. The main tokendrop pilot has 90/90 native and 90/90 one-chunk physical cases. Three ablation methods are complete at 90/90; random-basis has 59/90 cases and remains running.

All methods use the same 30 source episodes, clean k=0 controls, k=3/k=7 interference cases, held-out noise, and offline persistence protocol. Values are descriptive means over cases; no outcome threshold stops the queue.

| Method | Cases | Interfered | Repaired | Teacher | Negative | Random | Interpolation | Persistence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tokendrop-paired-one-step | 90 | +0.00000000 | +0.00145653 | +0.66666667 | -0.00097382 | +0.00000517 | +0.00041527 | +0.00013741 |
| tokendrop-centroid-two-step | 90 | +0.00000000 | +0.00102640 | +0.66666667 | -0.00048256 | -0.00037391 | +0.00016946 | +0.00028046 |
| tokendrop-ot-two-step | 90 | +0.00000000 | +0.00086045 | +0.66666667 | -0.00039872 | -0.00019752 | +0.00049735 | +0.00027736 |
| tokendrop-random-basis-two-step | 59 | +0.00000000 | +0.00144410 | +0.66101695 | -0.00174768 | +0.00009437 | +0.00008971 | +0.00009417 |

The native reports remain in the local preparation directory; `summary.json` records each source path and uncompressed SHA-256 for exact provenance. The complete primary raw exports are in `2026-09-11-tokendrop90`.

Remaining work: finish random-basis, run the frame-sampling ablations, and complete queued physical follow-ups.
