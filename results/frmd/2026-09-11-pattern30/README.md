# Completed PatternLock development block

Ten training-source episodes × clean k=0 and interference k=3/7 = 30 cases.
All use paired two-step FRMD with held-out acceptance and a history-difference
basis. Full original reports are gzip-compressed; their uncompressed hashes are
listed in summary.json. Neutral and negative outcomes are retained.

| k | FRMD recovery | Random | Negative | Interpolation | Offline persistence |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | +0.001208 | +0.001004 | -0.000945 | +0.001772 | +0.000913 |
| 7 | +0.001485 | +0.001779 | -0.003163 | +0.002880 | -0.000150 |

These descriptive action-space means show a small positive main effect, with
negative edits in the opposite direction. Random/interpolation controls also
improve, and interpolation has a larger mean. This does not establish FRMD
superiority. Offline persistence is mixed. No metric threshold stops the queue.

The accompanying physical analysis is a timestamped partial subset available
when this block completed, not all 30 physical evaluations. The GPU task block
was checkpointed before inserting engineering preflights; remaining tasks resume
from completed reports after those checks.
