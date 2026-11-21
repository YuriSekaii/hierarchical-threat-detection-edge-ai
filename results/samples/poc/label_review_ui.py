"""LabelImg-style OpenCV UI for human review of lab knife tile labels.

  python label_review_ui.py [--all-flagged] [--selftest]
"""
import argparse, csv, json, os, sys, tempfile, time
from collections import defaultdict

import cv2
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LAB_LABEL = os.path.join(REPO, "results", "stage1_kd", "lab_label")
TILE = 330
COLORS = ((0, 255, 0), (255, 255, 0), (255, 0, 255))
VIS = False  # --vis: how visible is the labelled knife (label_stage1_visibility.py, check_stage1_misses.py), keys c / b / h / x / u / w / m

CSV_COLS_BASE = ["tile_id", "img", "split", "take", "frame", "kind"]
CSV_COLS_TAIL = ["n_votes", "agree", "flag_reason", "claude_label", "claude_conf",
                 "human_label", "human_box", "final_label", "final_source", "last_update"]


def read_csv(path):
    rows = {}
    fields = []
    if not os.path.exists(path):
        return rows, fields
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fields = r.fieldnames or []
        for row in r:
            try:
                tid = int(row["tile_id"])
            except Exception:
                continue
            rows[tid] = row
    return rows, fields


def write_csv_atomic(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def format_vote(v):
    if v is None:
        return ""
    if isinstance(v, list):
        return json.dumps(v)
    return str(v)


def parse_vote_json(v):
    if v is None or v == "":
        return None
    if isinstance(v, (list, dict)):
        return v
    v = str(v)
    if v.startswith("["):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def compute_final(row, agree_label=""):
    human = row.get("human_label", "")
    if human not in ("", None):
        return (human, "human")
    if row.get("agree") in ("1", "True", "true", True) and agree_label:
        return (agree_label, "vote")
    cl = row.get("claude_label", "")
    try:
        cc = float(row.get("claude_conf", "") or 0)
    except Exception:
        cc = 0.0
    if cl not in ("", None) and cc >= 0.8:
        return (cl, "claude")
    if cl not in ("", None):
        return ("", "needs_human")
    return ("", "needs_claude")


def tile_box(it, v):
    if isinstance(v, str):
        return next(p["box"] for p in it["props"] if p["letter"] == v)
    s = it["crop"] / TILE
    return [it["x0"] + v[0] * s, it["y0"] + v[1] * s, it["x0"] + v[2] * s, it["y0"] + v[3] * s]


def screen_size():
    try:
        import ctypes
        u = ctypes.windll.user32
        u.SetProcessDPIAware()  # 1 window px = 1 screen px, no blurry Windows scaling
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        return 1366, 768


FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(img, txt, org, col, scale=0.6, th=1):
    # text on a dark backing so it reads on any background
    (w, h), base = cv2.getTextSize(txt, FONT, scale, th)
    x, y = org
    cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + base), (0, 0, 0), -1)
    cv2.putText(img, txt, (x, y), FONT, scale, col, th, cv2.LINE_AA)


def draw_zoom(img, it, size, extra_box=None):
    """Crop around the hand, scaled to size x size; boxes drawn after scaling so lines stay thin and sharp."""
    c = it["crop"]
    crop = img[it["y0"]:it["y0"] + c, it["x0"]:it["x0"] + c]
    zoom = cv2.resize(crop, (size, size), interpolation=cv2.INTER_CUBIC)
    s = size / c
    for p, col in zip(it.get("props", []), COLORS):
        b = p["box"]
        p1 = (int((b[0] - it["x0"]) * s) - 3, int((b[1] - it["y0"]) * s) - 3)
        p2 = (int((b[2] - it["x0"]) * s) + 3, int((b[3] - it["y0"]) * s) + 3)
        cv2.rectangle(zoom, p1, p2, col, 2)
        label(zoom, p.get("letter", ""), (max(2, p1[0]), max(22, p1[1] - 4)), col, 0.8, 2)
    if extra_box and len(extra_box) == 4:  # human box, stored in 330-px tile coords
        k = size / TILE
        cv2.rectangle(zoom, (int(extra_box[0] * k), int(extra_box[1] * k)),
                      (int(extra_box[2] * k), int(extra_box[3] * k)), (0, 0, 255), 2)
    return zoom


def draw_frame(img, it, max_w, max_h):
    """Whole frame scaled to fit max_w x max_h; yellow square = the zoomed area."""
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h)
    vis = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    x0, y0, c = it["x0"], it["y0"], it["crop"]
    cv2.rectangle(vis, (int(x0 * s), int(y0 * s)), (int((x0 + c) * s), int((y0 + c) * s)), (0, 255, 255), 2)
    return vis


def render(img, it, row, vcols, pos, total, extra_box=None, draw_mode=False, screen=(1366, 768)):
    """One window image sized to the screen: zoom on the left, frame and text on the right."""
    sw, sh = screen
    z = max(TILE, min(sh - 110, (sw - 30) // 2))  # leaves room for the title bar and taskbar
    rw = sw - 30 - z
    zoom = draw_zoom(img, it, z, extra_box)
    frame = draw_frame(img, it, rw, z * 0.56)
    canvas = np.full((z, z + rw, 3), 30, np.uint8)
    canvas[:, :z] = zoom
    canvas[:frame.shape[0], z:z + frame.shape[1]] = frame

    y = frame.shape[0] + 26
    x = z + 12
    def put(txt, col=(230, 230, 230), scale=0.55):
        nonlocal y
        cv2.putText(canvas, str(txt), (x, y), FONT, scale, col, 1, cv2.LINE_AA)
        y += 23
    put(f"{pos} / {total}    tile {row.get('tile_id', '')}    {row.get('split', '')}", (0, 255, 255), 0.65)
    put(os.path.basename(row.get("img", "")), (170, 170, 170), 0.45)
    put("Votes:  " + "   ".join(f"{vn} {row.get(vn, '') or '-'}" for vn in vcols))
    put(f"Claude: {row.get('claude_label', '') or '-'}  (conf {row.get('claude_conf', '') or '-'})")
    if row.get("human_label"):
        put(f"Your answer: {row['human_label']}", (80, 255, 80))
    if VIS:
        y += 6
        put("How well can you see the knife in the green box?", (255, 255, 0), 0.5)
        put("c  clear: you can tell it is a knife     b  blurred by motion", (255, 255, 0), 0.5)
        put("h  hidden / barely visible     x  no knife     u  cannot tell", (255, 255, 0), 0.5)
        put("w  knife clearly there, but the green box misses it", (255, 255, 0), 0.5)
        put("m  knife hidden / barely visible, and the green box misses it", (255, 255, 0), 0.5)
        if row.get("flag_reason"):
            put(row["flag_reason"], (0, 165, 255), 0.5)
    else:
        for p, col in zip(it.get("props", []), COLORS):
            sc = max(p.get("src", {}).values(), default=0)
            put(f"box {p.get('letter')}: best detector score {sc:.2f}", col, 0.5)
        y += 6
        put("a / b / c  this box has the knife     k  knife, no box fits", (255, 255, 0), 0.5)
        put("n  no knife     u  cannot tell     d  draw a box (drag on zoom)", (255, 255, 0), 0.5)
    put("space / right  next     p / left  back     z  undo     q  quit", (255, 255, 0), 0.5)
    if draw_mode:
        label(canvas, "DRAW MODE: drag a box on the zoom", (12, z - 16), (0, 0, 255), 0.8, 2)
    return canvas


def voter_columns(fields):
    skip = set(CSV_COLS_BASE + CSV_COLS_TAIL)
    return [f for f in fields if f not in skip]


def recompute_and_save(lab_label, rows_dict, fields):
    rows = [rows_dict[k] for k in sorted(rows_dict.keys())]
    write_csv_atomic(os.path.join(lab_label, "votes.csv"), rows, fields)


def run_ui(lab_label=LAB_LABEL, repo=REPO, all_flagged=False):
    rows_dict, fields = read_csv(os.path.join(lab_label, "votes.csv"))
    if not rows_dict:
        print("no votes.csv; run tally first")
        return
    items_path = os.path.join(lab_label, "lab_items.json")
    if not os.path.exists(items_path):  # review_stage1_candidates.py writes items.json
        items_path = os.path.join(lab_label, "items.json")
    items = {it["id"]: it for it in json.load(open(items_path, encoding="utf-8"))}

    vcols = voter_columns(fields)
    queue = []
    for tid in sorted(rows_dict.keys()):
        row = rows_dict[tid]
        fs = row.get("final_source", "")
        if all_flagged:
            if row.get("agree") not in ("1", "True", "true") and fs != "human":
                queue.append(tid)
        else:
            if fs == "needs_human":
                queue.append(tid)
    if not queue:
        print("no tiles to review")
        return

    # start at first unanswered (no human_label)
    idx = 0
    for i, tid in enumerate(queue):
        if not rows_dict[tid].get("human_label"):
            idx = i
            break

    history = []
    draw_state = {"active": False, "start": None, "box": None}
    window = "label_review_ui"
    screen = screen_size()
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)  # shown 1:1, never squashed
    cv2.moveWindow(window, 0, 0)  # sized to the screen, so it must start at the corner

    def mouse_cb(event, x, y, flags, param):
        if not draw_state["active"]:
            return
        # zoom panel is on right; we store coords relative to zoom image origin passed via param
        ox, oy, z = param.get("origin", (0, 0, 2.0))
        # convert window coords to zoom-tile coords
        tx = (x - ox) / z
        ty = (y - oy) / z
        tx = max(0, min(TILE - 1, tx))
        ty = max(0, min(TILE - 1, ty))
        if event == cv2.EVENT_LBUTTONDOWN:
            draw_state["start"] = (tx, ty)
            draw_state["box"] = None
        elif event == cv2.EVENT_MOUSEMOVE and draw_state["start"]:
            sx, sy = draw_state["start"]
            draw_state["box"] = [min(sx, tx), min(sy, ty), max(sx, tx), max(sy, ty)]
        elif event == cv2.EVENT_LBUTTONUP and draw_state["start"]:
            sx, sy = draw_state["start"]
            draw_state["box"] = [min(sx, tx), min(sy, ty), max(sx, tx), max(sy, ty)]
            draw_state["start"] = None

    mouse_param = {"origin": (0, 0, 2.0)}
    cv2.setMouseCallback(window, mouse_cb, mouse_param)

    def save_answer(tid, label, box=None):
        row = rows_dict[tid]
        prev = (row.get("human_label", ""), row.get("human_box", ""),
                row.get("final_label", ""), row.get("final_source", ""), row.get("last_update", ""))
        history.append((tid, prev))
        row["human_label"] = format_vote(label)
        row["human_box"] = json.dumps(box) if box else ""
        fl, fs = compute_final(row)
        row["final_label"] = format_vote(fl) if fl not in ("", None) else ""
        row["final_source"] = fs
        row["last_update"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        recompute_and_save(lab_label, rows_dict, fields)

    while 0 <= idx < len(queue):
        tid = queue[idx]
        row = rows_dict[tid]
        it = items.get(tid)
        if it is None:
            idx += 1
            continue
        img_path = os.path.join(repo, it["img"])
        img = cv2.imread(img_path)
        if img is None:
            idx += 1
            continue

        # current drawn box from human_box or draw_state
        hb = row.get("human_box", "")
        extra = None
        if hb:
            try:
                extra = json.loads(hb)
            except Exception:
                extra = None
        if draw_state.get("box"):
            extra = draw_state["box"]

        canvas = render(img, it, row, vcols, idx + 1, len(queue), extra, draw_state["active"], screen)
        mouse_param["origin"] = (0, 0, canvas.shape[0] / TILE)  # zoom sits at the top-left
        cv2.imshow(window, canvas)
        kx = cv2.waitKeyEx(30)  # waitKeyEx: arrow keys on Windows are 2424832 (left) and 2555904 (right)
        key = {2424832: 2, 2555904: 3, 65361: 2, 65363: 3}.get(kx, kx & 0xFF if kx >= 0 else 255)

        if key == ord('q'):
            break
        elif key in (ord('p'), 81, 2):  # p or left arrow
            idx = max(0, idx - 1)
            draw_state["active"] = False
        elif key in (ord(' '), 83, 3, ord('\r'), ord('\n')):  # space/right
            idx = min(len(queue) - 1, idx + 1)
            draw_state["active"] = False
        elif key == ord('z'):
            if history:
                htid, prev = history.pop()
                if htid in rows_dict:
                    hl, hb0, fl, fs, lu = prev
                    rows_dict[htid]["human_label"] = hl
                    rows_dict[htid]["human_box"] = hb0
                    rows_dict[htid]["final_label"] = fl
                    rows_dict[htid]["final_source"] = fs
                    rows_dict[htid]["last_update"] = lu
                    recompute_and_save(lab_label, rows_dict, fields)
                # jump to that tile if in queue
                if htid in queue:
                    idx = queue.index(htid)
            draw_state["active"] = False
        elif key == ord('d'):
            draw_state["active"] = True
            draw_state["start"] = None
            draw_state["box"] = None
        elif VIS and key in (ord('c'), ord('b'), ord('h'), ord('x'), ord('u'), ord('w'), ord('m')):
            save_answer(tid, chr(key), None)
            idx = min(len(queue) - 1, idx + 1)
        elif VIS:
            pass  # box keys mean nothing in the visibility review
        elif key in (ord('a'), ord('b'), ord('c')) and chr(key) not in [p.get("letter") for p in it.get("props", [])]:
            pass  # no suggestion with that letter on this tile
        elif key in (ord('a'), ord('b'), ord('c'), ord('k'), ord('n'), ord('u')):
            label = chr(key)
            save_answer(tid, label, None)
            draw_state["active"] = False
            idx = min(len(queue) - 1, idx + 1)
        elif draw_state["active"] and draw_state.get("box") and draw_state.get("start") is None:
            # mouse up finished a box
            box = [round(float(v), 1) for v in draw_state["box"]]
            if abs(box[2]-box[0]) > 2 and abs(box[3]-box[1]) > 2:
                save_answer(tid, "box", box)
                draw_state["active"] = False
                draw_state["box"] = None
                idx = min(len(queue) - 1, idx + 1)

    cv2.destroyAllWindows()
    print(f"reviewed queue of {len(queue)}; stopped at {idx+1}")


def selftest():
    td = tempfile.mkdtemp(prefix="label_review_ui_test_")
    try:
        lab = td
        # fake items
        items = [
            {"id": 0, "img": "dummy0.jpg", "split": "test", "take": "t0", "frame": 0, "x0": 0, "y0": 0, "crop": 330,
             "kind": "hand", "props": [{"letter": "a", "box": [10,10,50,50], "src": {"teacher": 0.5}}]},
            {"id": 1, "img": "dummy1.jpg", "split": "test", "take": "t1", "frame": 1, "x0": 0, "y0": 0, "crop": 330,
             "kind": "hand", "props": [{"letter": "b", "box": [20,20,60,60], "src": {"teacher": 0.3}}]},
        ]
        json.dump(items, open(os.path.join(lab, "lab_items.json"), "w"))
        fields = CSV_COLS_BASE + ["first", "haiku1"] + CSV_COLS_TAIL
        rows = []
        for it in items:
            row = {k: "" for k in fields}
            row.update({k: it.get(k, "") for k in CSV_COLS_BASE})
            row["tile_id"] = str(it["id"])
            row["first"] = "n"
            row["haiku1"] = "a"
            row["n_votes"] = "2"
            row["agree"] = "0"
            row["flag_reason"] = "class_mismatch"
            row["claude_label"] = "n"
            row["claude_conf"] = "0.6"
            row["final_label"] = ""
            row["final_source"] = "needs_human"
            rows.append(row)
        write_csv_atomic(os.path.join(lab, "votes.csv"), rows, fields)

        # merge/read/save logic
        rows_dict, fields2 = read_csv(os.path.join(lab, "votes.csv"))
        assert len(rows_dict) == 2
        assert fields2 == fields
        # simulate answer
        rows_dict[0]["human_label"] = "a"
        fl, fs = compute_final(rows_dict[0])
        rows_dict[0]["final_label"] = fl
        rows_dict[0]["final_source"] = fs
        rows_dict[0]["last_update"] = "2026-01-01T00:00:00"
        recompute_and_save(lab, rows_dict, fields2)
        rows2, _ = read_csv(os.path.join(lab, "votes.csv"))
        assert rows2[0]["human_label"] == "a"
        assert rows2[0]["final_source"] == "human"
        assert rows2[1]["human_label"] == ""
        assert rows2[1]["claude_label"] == "n"
        # merge keeps existing
        rows2[1]["claude_conf"] = "0.9"
        recompute_and_save(lab, rows2, fields2)
        rows3, _ = read_csv(os.path.join(lab, "votes.csv"))
        assert rows3[0]["human_label"] == "a"
        assert rows3[1]["claude_conf"] == "0.9"
        print("selftest ok")
    finally:
        # cleanup best-effort
        for root, dirs, files in os.walk(td, topdown=False):
            for n in files:
                try:
                    os.remove(os.path.join(root, n))
                except Exception:
                    pass
            for n in dirs:
                try:
                    os.rmdir(os.path.join(root, n))
                except Exception:
                    pass
        try:
            os.rmdir(td)
        except Exception:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-flagged", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--vis", action="store_true", help="visibility review: keys c / b / h / x / u")
    ap.add_argument("--dir", default=LAB_LABEL, help="folder with votes.csv and lab_items.json or items.json")
    args = ap.parse_args()
    VIS = args.vis
    if args.selftest:
        selftest()
    else:
        run_ui(lab_label=os.path.abspath(args.dir), all_flagged=args.all_flagged)
