# Knife label review: instructions for one reviewer agent

You check knife labels on tiles from a lab security camera. You only look at images and write JSON files. Do not run scripts, do not edit any other file.

## Input

Review sheets `results/stage1_kd/lab_label/review/review_NN.jpg` (repo root: `D:\Intern AI Project\hierarchical-threat-detection-edge-ai`). Each sheet is 1980x1980 px: 36 tiles in a 6x6 grid, each tile 330x330 px. Black slots are empty: skip them.

The red text at the top-left of a tile is `<tile id> <current label>`. Each tile is a close-up around the hand of the actor, or a smaller whole-person view. In most videos the actor holds a knife: a dark blade about 10-15 cm long with a dark handle, held in the fist.

Coloured rectangles are detector suggestions (`a` green, `b` cyan, `c` magenta), drawn a few pixels larger than the object. After the letter, `t.NN` is the teacher detector's score and `s.NN` the shipped detector's score. Scores are often wrong; judge by what you see.

Every tile here is one where the current label and the detectors disagree:
- label `n` but a detector scores a box high: is there really no knife, or did the first labeller miss it?
- label `a`/`b`/`c` but the detectors score that box low: is it really a knife?
- label `k`: a knife was seen but no rectangle fits it.

## Decision per tile

The correct label, using the same codes:
- `"a"`, `"b"` or `"c"`: that rectangle contains the knife (blade and/or handle). If two cover it, pick the tighter one that still covers the whole visible knife.
- `"k"`: a knife is clearly visible but no rectangle covers it.
- `"n"`: no knife in the tile (empty hand, no person, or the rectangle is on something else).
- `"u"`: cannot tell (hand cut off, hidden, or too blurred).

Not knives (answer `"n"` for a rectangle on them): a grey knife-shaped mark on the white cart/shelf in the background, chair wheels and legs, table legs, cables, floor stripes, door handles, trouser folds, sleeve ends, shoes, watches, fingers of an empty hand, a phone, wallet, pen or remote.

A knife must be visible as a blade or handle in the hand. Do not guess from the arm pose alone. If you hesitate between a letter and `"u"`, choose `"u"`.

## Output

For each sheet you are given, in order:
1. Read the sheet image once with the Read tool.
2. Immediately write `results/stage1_kd/lab_label/review_fix/review_NN.json` (same number as the sheet) with the Write tool, one entry for **every** tile on that sheet, e.g. `{"812": "a", "903": "n", "1150": "u"}`, including tiles where you keep the current label.

Do not re-read sheets, do not skip tiles. Your final reply: the sheets done, how many labels you changed, and from what to what (e.g. `n -> a: 12`).
