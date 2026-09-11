# Second memory mechanism and real RGB preflight

The released frame-sampling checkpoint completed the same first k=3 case from
each task, with the same 30-episode prepared data and source manifest.

| Task | FRMD recovery | Offline persistence | Random | Negative | Interpolation |
| --- | ---: | ---: | ---: | ---: | ---: |
| PatternLock | 0 | 0 | 0 | 0 | 0 |
| RouteStick | +0.000112 | +0.000669 | +0.000057 | +0.000090 | +0.000096 |
| VideoPlaceButton | +0.002968 | +0.001296 | -0.002137 | -0.001568 | +0.003765 |

These are descriptive action-space results from three development cases. Each
also completed six physical conditions × eight matched noise replicas. In the
PatternLock case, all conditions averaged 0.75 subgoal advancement per chunk;
RouteStick and VideoPlaceButton all averaged zero. FRMD therefore has zero
relative physical advancement gain in these three cases. Every chunk remained
ongoing; this is not an episode-level failure rate. Full unchanged reports and
portable analysis are included.

The tokendrop VideoPlaceButton repair also completed a real RGB engineering
preflight: two queries each for interfered and repaired branches, 40 actions per
branch, identical initial physical and RGB/state fingerprints. The second query
encoded 20 newly executed frames. Of 512 source tokens, 505 survived the update;
the applied repair norm changed from 0.901500 to 0.898497 through token eviction.
No teacher retrieval/refit occurred in the repaired branch. Both branches remained
ongoing with zero subgoal advancement. This verifies the live update/edit path,
not a physical performance advantage. The six-condition, ten-query experiments
remain in the queue.

A fresh clone of GitHub commit d5b9130 passed 145 tests, compileall, all seven
new CLI help commands and the default dry-run. Preparation recipes are archived
under reproduction/ with their original execution paths. The full experiment
queue resumed after the preflights; no result-based stop threshold was applied.
