import os
import sys

import numpy as np

sys.path[:0] = [os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))]
import generate_from_edits as M  # noqa: E402


def test_directions_go_front_right_back_left():
    assert M.side_of("front", "right") == 1 and M.side_of("right", "back") == 1 and M.side_of("left", "front") == 1
    assert M.side_of("front", "left") == -1 and M.side_of("back", "right") == -1


def _take(cut=False, strikes=((60, 70, 80), (120, 130, 140))):
    return dict(strikes=list(strikes), trips=list(strikes), any_trips=list(strikes), offset=20, cut=cut)


def _walk(t, n=300, missing=()):
    keep = M.full_take
    gone = np.zeros(n, bool)
    gone[list(missing)] = True
    M.full_take = lambda _t: (np.arange(n, dtype=np.float32), 0, gone)
    try:
        return [int(w[0]) for w in M.walk_in_windows(t)]
    finally:
        M.full_take = keep


def test_normal_windows_stay_outside_the_strikes():
    # trips at 80..100 and 140..160 in take frames, with a margin of 15
    assert _walk(_take()) == [0, 10] + list(range(180, 251, 10))


def test_a_take_cut_short_gives_nothing_after_its_last_trip():
    assert _walk(_take(cut=True)) == [0, 10]


def test_a_take_with_no_strikes_gives_nothing():
    assert _walk(_take(strikes=())) == []


def test_windows_mostly_undetected_are_skipped():
    assert 200 not in _walk(_take(), missing=range(210, 225))  # 15 frames gone
    assert 200 in _walk(_take(), missing=range(210, 220))  # 10 frames is still allowed


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
