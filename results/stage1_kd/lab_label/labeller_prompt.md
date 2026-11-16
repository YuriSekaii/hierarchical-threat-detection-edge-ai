# Knife labelling: instructions for one labeller agent

You label knife tiles from a lab security camera. You only look at images and write JSON files. Do not run scripts, do not edit any other file.

## Input

Sheets `results/stage1_kd/lab_label/sheets/sheet_NNNN.jpg` (repo root: `D:\Intern AI Project\hierarchical-threat-detection-edge-ai`). Each sheet is 1320x1320 px: 16 tiles in a 4x4 grid, each tile 330x330 px. The red number at the top-left of a tile is its **tile id**. The last sheet may have black empty slots: skip them.

Each tile is a close-up crop around the hand of one person (the actor). In most videos the actor holds a knife: a dark blade about 10-15 cm long with a dark handle, held in the fist, blade pointing out, down or up. In some videos the hand is empty or holds nothing special.

Coloured rectangles with a letter are **suggestions from a detector**; they are often wrong:
- `a` = green, `b` = cyan, `c` = magenta.
- The rectangle is drawn a few pixels larger than the detected object.

## Decision per tile

One of:
- `"a"`, `"b"` or `"c"`: this rectangle contains the knife (blade and/or handle). If two rectangles both cover the same knife, pick the one that fits it tighter while still covering the whole visible knife.
- `"k"`: a knife is clearly visible in the tile but no rectangle covers it.
- `"n"`: no knife in this tile. Use this when the hand is empty, or a rectangle sits on something that is not a knife.
- `"u"`: cannot tell. The hand is cut off or hidden, or it is so blurred that you cannot say whether a knife is there.

A tile with no person at all in it (empty room, furniture, floor only) is `"n"`, not `"u"`.

Some tiles show a whole person, smaller (the crop is around the body, not the hand). Judge them the same way: look at both hands for a knife.

Things the detector confuses with a knife (answer `"n"` for a rectangle on them):
- a grey knife-shaped mark or shadow on the white metal cart / shelf in the background, at the same place in many tiles;
- chair wheels, chair legs, table legs, cables, floor stripes, door handles;
- folds of dark trousers, a sleeve end, a shoe, a watch, fingers of an empty hand;
- a phone, wallet, pen or remote control in the hand.

Be strict: a knife must be visible as a blade or a knife handle in the hand. Do not guess from the arm pose alone. If you hesitate between a letter and `"u"`, choose `"u"`.

## Output

For each sheet you are given, in order:
1. Read the sheet image once with the Read tool.
2. Immediately write `results/stage1_kd/lab_label/decisions/sheet_NNNN.json` (same number as the sheet) with the Write tool, one entry for **every** tile id on that sheet, e.g. `{"32": "a", "33": "n", "34": "u", "35": "k"}`.

Do not re-read sheets, do not skip tiles, do not stop before the last sheet. Your final reply: the sheet range done and how many tiles got a letter, `k`, `n`, `u`.
