# Modification Report: why Stage 1 scores 86.9 % at 640x640 and 76.4 % at 384x640 (2026-11-14)

**Status: measurement only. No source file, engine or weight was changed.** The pipeline still runs at 384x640.

## 1. Question

The shipped detector `weights/yolo_weapon_distilled.pt` scores F1 86.9 % at 640x640 and 76.4 % at 384x640 on the 715-image ground-truth set. A 640x640 input of a wide image is mostly padding, so it should not help. Where do the 10 points go, and should the pipeline move to another input size?

## 2. What was run

```bash
.venv/Scripts/python.exe scripts/validate_stage1_ground_truth.py --weights weights/yolo_weapon_distilled.pt --imgsz 416 640
```

Repeated for the sizes below, confidence 0.45. The test images are 2560x1600 (16:10). The shrink factor is `min(H / 1600, W / 2560)`.

For the lab camera: a one-off script ran the detector on Nov 5 knife takes 002, 005, 010, 014, 018, 022 and phone takes 006, 008 (every 6th frame, confidence 0.45) with the frame shrunk four ways and pasted unscaled into a 384x640 input.

## 3. Results on the ground-truth set

| Input size | Shrink factor | Precision | Recall | F1 |
|---|---|---|---|---|
| 384x640 (deployed) | 0.240 | 78.8 % | 74.2 % | 76.4 % |
| 384x672 | 0.240 | 78.8 % | 74.2 % | 76.4 % |
| 384x608 | 0.2375 | 79.3 % | 76.7 % | 78.0 % |
| 416x608 | 0.2375 | 79.7 % | 76.7 % | 78.1 % |
| 400x640 | 0.250 | 88.2 % | 84.2 % | 86.1 % |
| 416x640 | 0.250 | 88.2 % | 84.2 % | 86.1 % |
| 480x640 | 0.250 | 89.0 % | 84.2 % | 86.5 % |
| 640x640 | 0.250 | 89.8 % | 84.2 % | 86.9 % |
| 448x704 | 0.275 | 75.5 % | 73.3 % | 74.4 % |
| 512x864 | 0.3375 | 67.5 % | 75.4 % | 71.3 % |
| 768x1280 | 0.480 | 67.7 % | 61.3 % | 64.3 % |

- Every size that shrinks the image by exactly 0.25 gives recall 84.2 % and F1 86–87 %. 640x640 is the same pixels as 416x640 plus padding.
- Every other factor is about 10 F1 points lower, including larger inputs. The score does not rise with resolution.
- 384x640 gives factor 0.24 only because the test images are 16:10; the image height, not the width, limits the fit.

Knife size at the deployed input (240 knives): median 26 px long and 17 px wide, none longer than 48 px.

| Knife long side at 384x640 | Knives | Recall at 384x640 | Recall at 640x640 |
|---|---|---|---|
| under 16 px | 5 | 60 % | 100 % |
| 16–24 px | 84 | 75 % | 86 % |
| 24–32 px | 92 | 79 % | 88 % |
| 32–48 px | 59 | 75 % | 78 % |

## 4. Results on the lab camera

Share of frames with a detection. The lab camera is 2560x1440 (16:9); the pipeline analyses the 1920x1440 centre crop, which a 384x640 input shrinks by 0.267.

| Take | 4:3 crop as fed today (0.267) | 4:3 crop at 0.25 | Full 16:9 frame (0.25) | 4:3 crop at 0.30 |
|---|---|---|---|---|
| 002 knife | 88 % | 81 % | 81 % | 83 % |
| 005 knife | 74 % | 76 % | 77 % | 74 % |
| 010 knife | 66 % | 63 % | 64 % | 65 % |
| 014 knife | 35 % | 34 % | 35 % | 39 % |
| 018 knife | 41 % | 45 % | 41 % | 42 % |
| 022 knife | 62 % | 54 % | 62 % | 58 % |
| 006 phone | 61 % | 58 % | 57 % | 55 % |
| 008 phone | 19 % | 21 % | 19 % | 16 % |

No factor is better than another on the lab camera.

## 5. Conclusion

- The 10-point gap is a property of the test set, not of the input resolution. The detector is tuned to how the test-set images look after a 4x shrink, which suggests those images are close to its training images.
- 416x640 is the right size for scoring this 16:10 test set. It is not better for the live camera: a 16:9 frame already gets factor 0.25 in a 384x640 input, and the measured lab takes show no gain at any factor.
- Moving the pipeline to 416x640 would need new TensorRT engines for the detector and the pose model on both PCs and about 8 % more pixels per frame, for no measured gain on the camera. It was not done.
- The 86.9 % is an upper figure for images like the test set. The lab-camera detection share (36–88 % of frames) is the figure that describes the deployed system.

## 6. Not checked

- Whether the ground-truth images overlap the original training set; that set is no longer on disk.
- The cause of the sensitivity (object size or the resampling pattern of an exact 4x shrink).
- TensorRT engines; all runs used the PyTorch weights.
- The comparison in `MODIFICATION_REPORT_STAGE1_MODEL_COMPARISON.md` was measured at 384x640; a re-run of its main rows at 416x640 is recorded there.
