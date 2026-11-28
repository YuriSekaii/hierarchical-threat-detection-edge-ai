"""Unit tests for src/gpu_preprocess.py."""

import os
import sys
import unittest
import numpy as np
import torch

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.gpu_preprocess import letterbox_bgr_to_tensor, unletterbox_boxes, unletterbox_keypoints


class TestGpuPreprocess(unittest.TestCase):
    def _devices(self):
        devs = ["cpu"]
        if torch.cuda.is_available():
            devs.append("cuda:0")
        return devs

    def test_shapes_and_geometry(self):
        shapes = [
            (1440, 1920, 3),
            (720, 1280, 3),
            (2160, 3840, 3),
            (360, 640, 3),
            (720, 406, 3),
        ]
        for dev in self._devices():
            for h, w, c in shapes:
                frame = np.zeros((h, w, c), dtype=np.uint8)
                frame[:, :, 0] = 10   # B
                frame[:, :, 1] = 100  # G
                frame[:, :, 2] = 200  # R
                t, meta = letterbox_bgr_to_tensor(frame, imgsz=(384, 640), device=dev)
                self.assertEqual(tuple(t.shape), (1, 3, 384, 640))
                self.assertEqual(t.dtype, torch.float32)
                self.assertGreaterEqual(float(t.min()), 0.0)
                self.assertLessEqual(float(t.max()), 1.0)
                r, pad_x, pad_y, oh, ow = meta
                self.assertEqual((oh, ow), (h, w))
                # Center pixel should be RGB = (200, 100, 10) / 255
                cy, cx = 192, 320
                rgb = t[0, :, cy, cx].cpu().numpy()
                np.testing.assert_allclose(rgb, [200.0 / 255.0, 100.0 / 255.0, 10.0 / 255.0], atol=1e-3)
                # Padding border when aspect differs from 384x640 (e.g. 1440x1920 has left/right pad)
                if pad_x >= 2:
                    pad_px = t[0, :, 192, 0].cpu().numpy()
                    np.testing.assert_allclose(pad_px, [114.0 / 255.0] * 3, atol=1e-4)

    def test_non_contiguous_4_3_slice(self):
        full = np.full((1440, 2560, 3), 50, dtype=np.uint8)
        crop = full[:, 320:2240]  # 1920x1440 non-contiguous
        self.assertFalse(crop.flags["C_CONTIGUOUS"])
        for dev in self._devices():
            t, meta = letterbox_bgr_to_tensor(crop, imgsz=(384, 640), device=dev)
            self.assertEqual(tuple(t.shape), (1, 3, 384, 640))
            self.assertEqual(meta[3:], (1440, 1920))

    def test_unletterbox_roundtrip_and_missing_kpts(self):
        h, w = 1440, 1920
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        _, meta = letterbox_bgr_to_tensor(frame, imgsz=(384, 640), device="cpu")
        r, pad_x, pad_y, _, _ = meta

        orig_box = [200.0, 300.0, 500.0, 900.0]
        lb_box = [orig_box[0] * r + pad_x, orig_box[1] * r + pad_y, orig_box[2] * r + pad_x, orig_box[3] * r + pad_y]
        rec_box = unletterbox_boxes([lb_box], meta)[0]
        np.testing.assert_allclose(rec_box, orig_box, atol=1e-3)

        kpts = np.zeros((17, 2), dtype=np.float32)
        kpts[0] = [960.0 * r + pad_x, 720.0 * r + pad_y]
        kpts[5] = [0.0, 0.0]  # missing keypoint must stay (0, 0)
        rec_kpts = unletterbox_keypoints(kpts, meta)
        np.testing.assert_allclose(rec_kpts[0], [960.0, 720.0], atol=1e-3)
        np.testing.assert_allclose(rec_kpts[5], [0.0, 0.0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
