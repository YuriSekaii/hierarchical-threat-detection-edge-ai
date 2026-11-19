# kdgtNL: wrist-crop second pass, lab test (2842 frames, 1376 boxes)

| variant | 0.40: P / R | false boxes | 0.40 + gate: P / R | best R at P >= 0.90 (conf) | best R at P >= 0.90, gate (conf) |
|---|---|---|---|---|---|
| full frame | 90.5 / 79.1 | 114 | 90.4 / 77.7 | 79.1 % (0.40) | 77.7 % (0.40) |
| full + wrist crops 2x | 89.0 / 83.3 | 142 | 88.9 / 81.7 | 81.7 % (0.50) | 80.3 % (0.50) |
| wrist crops 2x only | 91.8 / 74.8 | 92 | 91.9 / 73.9 | 76.1 % (0.35) | 75.2 % (0.35) |
| full + wrist crops 4x | 85.8 / 80.4 | 183 | 85.9 / 78.9 | 75.7 % (0.55) | 74.4 % (0.55) |
| wrist crops 4x only | 84.5 / 20.2 | 51 | 85.8 / 20.2 | 7.3 % (0.60) | 7.3 % (0.60) |

| full-frame miss cause | boxes | found by wrist crops 2x at 0.40 | 4x at 0.40 |
|---|---|---|---|
| unseen | 126 | 11 | 8 |
| low_score | 140 | 45 | 10 |
| bad_box | 17 | 6 | 0 |
