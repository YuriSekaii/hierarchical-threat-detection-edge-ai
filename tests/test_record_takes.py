"""The recorder saves the pipeline's 4:3 crop, at the size given to the video file."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from record_takes import TakeWriter, crop_4_3


def test_crop_keeps_the_centre_of_wide_frames():
    frame = np.zeros((1440, 2560, 3), np.uint8)
    frame[:, 320] = 255  # first column inside the crop
    crop = crop_4_3(frame)
    assert crop.shape == (1440, 1920, 3)
    assert crop[0, 0, 0] == 255 and crop[0, 1, 0] == 0


def test_crop_leaves_4_3_and_narrower_frames_alone():
    for shape in ((1440, 1920, 3), (480, 480, 3)):
        assert crop_4_3(np.zeros(shape, np.uint8)).shape == shape


def test_writer_saves_the_cropped_size(tmp_path):
    path = str(tmp_path / "take.avi")
    writer = TakeWriter(path, (640, 480))
    for _ in range(5):
        writer.push(np.zeros((480, 854, 3), np.uint8))
    assert writer.finish()[1] == 5
    cap = cv2.VideoCapture(path)
    size = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames = 0
    while cap.read()[0]:
        frames += 1
    cap.release()
    assert size == (640, 480) and frames == 5
