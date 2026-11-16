# Knife label check: instructions for one Claude checker

You settle knife labels that three AI labellers disagreed on. You only look at images and write JSON files. Do not run scripts, do not edit any other file.

## Input

Check sheets `results/stage1_kd/lab_label/claude_check/sheets/check_NN.jpg` (repo root `D:\Intern AI Project\hierarchical-threat-detection-edge-ai`). Each sheet is 1980x1980 px: 36 tiles in a 6x6 grid, each 330x330 px. Black slots are empty: skip them. The tile ids of each sheet are listed in `results/stage1_kd/lab_label/claude_check/check_items.json` under the key `check_NN`, in reading order (left to right, top to bottom). Use that list for the ids; the red number at the top-left of each tile is the same id.

Each tile is a close-up around the hand of the actor from a lab security camera (some tiles show a whole person, smaller). In most videos the actor holds a knife: a dark blade about 10-15 cm long with a dark handle, held in the fist; it is often only 20-40 px long in the tile and blurred. Coloured rectangles are detector suggestions: `a` green, `b` cyan, `c` magenta. The small red text at the bottom of a tile shows the three earlier votes (`f:` first pass, `haiku1:`, `haiku2:`): they disagree, so do not follow them; judge the picture.

## Decision per tile

- `"a"`, `"b"` or `"c"`: that rectangle contains the knife (blade and/or handle). If two cover it, pick the tighter one that still covers the whole visible knife.
- `"k"`: a knife is clearly visible but no rectangle covers it.
- `"n"`: no knife (empty hand, no person, or the rectangle is on something else: chair legs or wheels, the grey knife-shaped mark on the white cart, trouser folds, sleeve ends, watches, a phone or remote).
- `"u"`: cannot tell (hand cut off, hidden, or too blurred).

With each decision give a **confidence from 0 to 1** that it is right: 0.9+ only when the blade or handle is plainly visible (or plainly absent from a clearly visible empty hand); 0.6-0.8 when it is likely but small or blurred; below 0.6 when you are guessing. Be honest: tiles below 0.6 go to a person, which is the point of this check.

## Output

For each sheet, in order: read the image once, then immediately write `results/stage1_kd/lab_label/claude_check/answers/check_NN.json` with one entry per tile id of that sheet: `{"247": ["b", 0.6], "456": ["n", 0.85], ...}`. Do not open other answer files. Final reply: the sheets done and how many tiles below 0.6 confidence.
