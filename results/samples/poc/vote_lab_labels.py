"""Multi-voter tally/review workflow for lab knife tile labels.

  python vote_lab_labels.py tally  -> votes.csv, claude_check sheets
  python vote_lab_labels.py claude -> merge claude_check/answers into votes.csv
  python vote_lab_labels.py final  -> lab_decisions_final.json
"""
import csv, glob, json, os, sys, tempfile
from collections import defaultdict

import cv2
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LAB_LABEL = os.path.join(REPO, "results", "stage1_kd", "lab_label")
CROP, TILE = 400, 330
COLORS = ((0, 255, 0), (255, 255, 0), (255, 0, 255))


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def draw(img, it, labels=None):
    s = TILE / it["crop"]
    tile = cv2.resize(img[it["y0"]:it["y0"] + it["crop"], it["x0"]:it["x0"] + it["crop"]], (TILE, TILE))
    for p, col in zip(it.get("props", []), COLORS):
        b = p["box"]
        p1 = (int((b[0] - it["x0"]) * s) - 4, int((b[1] - it["y0"]) * s) - 4)
        p2 = (int((b[2] - it["x0"]) * s) + 4, int((b[3] - it["y0"]) * s) + 4)
        cv2.rectangle(tile, p1, p2, col, 1)
        txt = labels(p) if labels else p.get("letter", "")
        cv2.putText(tile, txt, (p1[0], max(14, p1[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    return tile


def tile_box(it, v):
    if isinstance(v, str):
        return next(p["box"] for p in it["props"] if p["letter"] == v)
    s = it["crop"] / TILE
    return [it["x0"] + v[0] * s, it["y0"] + v[1] * s, it["x0"] + v[2] * s, it["y0"] + v[3] * s]


def write_sheets(tiles, cols, path_fmt, first=0):
    per = cols * cols
    for n in range(0, len(tiles), per):
        t = tiles[n:n + per]
        t = t + [np.zeros((TILE, TILE, 3), np.uint8)] * (-len(t) % cols)
        cv2.imwrite(path_fmt % (first + n // per), np.vstack([np.hstack(t[i:i + cols]) for i in range(0, len(t), cols)]))
    return -(-len(tiles) // per) if tiles else 0


def load_items(lab_label=LAB_LABEL):
    path = os.path.join(lab_label, "lab_items.json")
    items = json.load(open(path, encoding="utf-8"))
    return {it["id"]: it for it in items}, items


def load_json_dir(d):
    dec = {}
    if not os.path.isdir(d):
        return dec
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            data = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        for k, v in data.items():
            try:
                dec[int(k)] = v
            except Exception:
                pass
    return dec


def load_first(lab_label=LAB_LABEL):
    dec = load_json_dir(os.path.join(lab_label, "decisions"))
    fix = os.path.join(lab_label, "lab_decisions_fix.json")
    if os.path.exists(fix):
        dec.update({int(k): v for k, v in json.load(open(fix, encoding="utf-8")).items()})
    return dec


def load_voters(lab_label=LAB_LABEL):
    votes_dir = os.path.join(lab_label, "votes")
    voters = {}
    if not os.path.isdir(votes_dir):
        return voters
    for name in sorted(os.listdir(votes_dir)):
        d = os.path.join(votes_dir, name)
        if os.path.isdir(d):
            voters[name] = load_json_dir(d)
    return voters


def normalize_vote(vote, item):
    """Return ('k'/'n'/'u', None) or ('box', full_frame_box_xyxy)."""
    if vote is None:
        return (None, None)
    if isinstance(vote, list):
        if len(vote) == 4:
            return ("box", tile_box(item, vote))
        return (None, None)
    v = str(vote)
    if v in ("k", "n", "u"):
        return (v, None)
    if v in "abc":
        try:
            return ("box", tile_box(item, v))
        except Exception:
            return (None, None)
    return (None, None)


def vote_class_str(vote, item):
    cls, box = normalize_vote(vote, item)
    if cls is None:
        return ""
    if cls == "box":
        return "box"
    return cls


def same_class(votes_norm):
    # votes_norm: list of (cls, box) for voters that voted
    if not votes_norm:
        return False, "no_votes"
    cls0, box0 = votes_norm[0]
    for cls, box in votes_norm[1:]:
        if cls != cls0:
            return False, "class_mismatch"
    if cls0 == "box":
        boxes = [b for _, b in votes_norm if b is not None]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if iou(boxes[i], boxes[j]) < 0.3:
                    return False, "box_iou"
    return True, ""


CSV_COLS_BASE = ["tile_id", "img", "split", "take", "frame", "kind"]
CSV_COLS_TAIL = ["n_votes", "agree", "flag_reason", "claude_label", "claude_conf",
                 "human_label", "human_box", "final_label", "final_source", "last_update"]


def read_csv(path):
    rows = {}
    if not os.path.exists(path):
        return rows, []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                tid = int(row["tile_id"])
            except Exception:
                continue
            rows[tid] = row
    return rows, list(r.fieldnames) if 'r' in locals() else []


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


def format_vote(v):
    if v is None:
        return ""
    if isinstance(v, list):
        return json.dumps(v)
    return str(v)


def short_vote(v):
    if v is None:
        return ""
    if isinstance(v, list):
        return "box"
    return str(v)


def compute_final(row, agree_label, claude_checked, claude_majority=False):
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
    # Claude decides alone at >= 0.8, or at >= 0.6 when at least two of the three voters say the same
    if cl not in ("", None) and (cc >= 0.8 or (cc >= 0.6 and claude_majority)):
        return (cl, "claude")
    if claude_checked:
        return ("", "needs_human")
    return ("", "needs_claude")


def agreed_label_from_votes(votes_list):
    # votes_list: list of raw votes (same class assumed)
    if not votes_list:
        return ""
    v = votes_list[0]
    if isinstance(v, list):
        return json.dumps(v)
    # if letter, keep letter (so we know which box) - but all must agree same letter? 
    # normalize_vote maps a/b/c to box class but for agreement we checked class+IoU.
    # better return canonical: if they all picked same letter return it, else return first as box coords? spec says "agreed label"
    return str(v)


def tally(lab_label=LAB_LABEL, repo=REPO):
    items_by_id, items = load_items(lab_label)
    first = load_first(lab_label)
    voters = load_voters(lab_label)
    voter_names = list(voters.keys())
    existing, _ = read_csv(os.path.join(lab_label, "votes.csv"))

    rows = []
    counts = {"tiles": 0, "agree": 0}
    flag_reasons = defaultdict(int)
    needs_claude = []

    for it in items:
        tid = it["id"]
        prev = existing.get(tid, {})
        row = {c: "" for c in CSV_COLS_BASE + voter_names + CSV_COLS_TAIL}
        row.update({k: it.get(k, "") for k in CSV_COLS_BASE})
        row["tile_id"] = tid
        # preserve claude/human
        for k in ("claude_label", "claude_conf", "human_label", "human_box", "last_update"):
            if prev.get(k):
                row[k] = prev[k]

        votes = {"first": first.get(tid)}
        for vn in voter_names:
            votes[vn] = voters[vn].get(tid)
        row["first"] = format_vote(votes["first"]) if "first" in voter_names or True else ""
        # store per voter columns
        all_names = ["first"] + voter_names
        for vn in all_names:
            row[vn] = format_vote(votes.get(vn))

        # determine agreement among those who voted
        voted = [(vn, votes[vn]) for vn in all_names if votes.get(vn) is not None]
        norms = []
        for vn, v in voted:
            nrm = normalize_vote(v, it)
            if nrm[0] is not None:
                norms.append(nrm)
        agree, reason = same_class(norms) if norms else (False, "no_votes")
        if not norms:
            agree, reason = False, "no_votes"
        row["n_votes"] = str(len(norms))
        row["agree"] = "1" if agree else "0"
        row["flag_reason"] = "" if agree else reason

        # agreed label: if agree, pick canonical
        agree_label = ""
        if agree and norms:
            # prefer letter if all raw are same letter
            raw = [votes[vn] for vn, _ in voted if normalize_vote(votes[vn], it)[0] is not None]
            if raw and all(str(r) == str(raw[0]) for r in raw) and not isinstance(raw[0], list):
                agree_label = str(raw[0])
            elif raw and all(isinstance(r, list) for r in raw) and all(r == raw[0] for r in raw):
                agree_label = json.dumps(raw[0])
            else:
                # all box class but maybe different letters -> use first raw
                agree_label = format_vote(raw[0]) if raw else ""

        claude_checked = bool(row.get("claude_label"))
        fl, fs = compute_final(row, agree_label, claude_checked)
        row["final_label"] = format_vote(fl) if fl not in ("", None) else ""
        row["final_source"] = fs

        counts["tiles"] += 1
        if agree:
            counts["agree"] += 1
        else:
            flag_reasons[reason] += 1
        if row["final_source"] == "needs_claude":
            needs_claude.append((it, row, votes))
        rows.append(row)

    fieldnames = CSV_COLS_BASE + ["first"] + voter_names + CSV_COLS_TAIL
    # ensure unique order
    seen = set()
    fn = []
    for f in fieldnames:
        if f not in seen:
            fn.append(f)
            seen.add(f)
    write_csv_atomic(os.path.join(lab_label, "votes.csv"), rows, fn)

    # build claude check sheets
    def _prio(entry):  # hard conflicts first: knife vs no knife, then box mismatch, then knife vs unsure, then n vs u
        it, row, votes = entry
        cl = {vote_class_str(v, it) for v in votes.values() if v is not None} - {""}
        knife = bool(cl & {"box", "k"})
        return (0 if knife and "n" in cl else 1 if row["flag_reason"] == "box_iou" or cl == {"box", "k"} else 2 if knife else 3, it["id"])
    needs_claude.sort(key=_prio)
    if needs_claude:
        ck_dir = os.path.join(lab_label, "claude_check", "sheets")
        os.makedirs(ck_dir, exist_ok=True)
        tiles = []
        ids = []
        check_map = {}
        for it, row, votes in needs_claude:
            img = cv2.imread(os.path.join(repo, it["img"]))
            if img is None:
                # blank placeholder
                t = np.zeros((TILE, TILE, 3), np.uint8)
            else:
                t = draw(img, it)
            # red id
            cv2.putText(t, str(it["id"]), (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            # votes text
            parts = []
            for vn in ["first"] + voter_names:
                v = votes.get(vn)
                if v is not None:
                    parts.append(f"{'f' if vn=='first' else vn}:{short_vote(v)}")
            vtxt = " ".join(parts)
            cv2.putText(t, vtxt[:28], (3, TILE - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            tiles.append(t)
            ids.append(it["id"])
        n = write_sheets(tiles, 6, os.path.join(ck_dir, "check_%02d.jpg"))
        for k in range(n):
            check_map[f"check_{k:02d}"] = ids[k * 36:(k + 1) * 36]
        json.dump(check_map, open(os.path.join(lab_label, "claude_check", "check_items.json"), "w", encoding="utf-8"))

    print(f"tiles: {counts['tiles']}, agree: {counts['agree']}")
    for r, c in sorted(flag_reasons.items()):
        print(f"  flagged {r}: {c}")
    print(f"needs_claude: {len(needs_claude)}")


def claude_mode(lab_label=LAB_LABEL, repo=REPO):
    # read claude_check/answers/*.json -> {tile id: [label, confidence 0..1]}
    answers = {}
    ans_dir = os.path.join(lab_label, "claude_check", "answers")
    for p in sorted(glob.glob(os.path.join(ans_dir, "*.json"))):
        try:
            data = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        for k, v in data.items():
            try:
                tid = int(k)
            except Exception:
                continue
            if isinstance(v, list) and len(v) == 2:
                answers[tid] = (v[0], v[1])
            elif isinstance(v, dict):
                answers[tid] = (v.get("label"), v.get("confidence", v.get("conf", 0)))
            else:
                answers[tid] = (v, 1.0)

    existing, fields = read_csv(os.path.join(lab_label, "votes.csv"))
    if not existing:
        # run tally logic context? need to create
        tally(lab_label, repo)
        existing, fields = read_csv(os.path.join(lab_label, "votes.csv"))

    items_by_id, _ = load_items(lab_label)
    voters = load_voters(lab_label)
    voter_names = list(voters.keys())
    first = load_first(lab_label)
    rows = []
    for tid in sorted(existing.keys()):
        row = existing[tid]
        if tid in answers:
            lbl, conf = answers[tid]
            row["claude_label"] = format_vote(lbl)
            row["claude_conf"] = str(conf)
        # recompute final
        # need agree_label again
        it = items_by_id.get(tid)
        agree_label = ""
        if row.get("agree") in ("1", "True", "true"):
            all_names = ["first"] + voter_names
            raw = []
            if it:
                for vn in all_names:
                    rv = parse_vote_json(row.get(vn, ""))
                    if rv is not None and normalize_vote(rv, it)[0] is not None:
                        raw.append(rv)
            if raw and all(str(r) == str(raw[0]) for r in raw) and not isinstance(raw[0], list):
                agree_label = str(raw[0])
            elif raw and all(isinstance(r, list) for r in raw) and all(r == raw[0] for r in raw):
                agree_label = json.dumps(raw[0])
            elif raw:
                agree_label = format_vote(raw[0])
        claude_checked = bool(row.get("claude_label"))
        claude_majority = False
        if it and claude_checked:
            cn = normalize_vote(parse_vote_json(row["claude_label"]), it)
            others = [normalize_vote(parse_vote_json(row.get(vn, "")), it) for vn in ["first"] + voter_names]
            claude_majority = cn[0] is not None and sum(
                o[0] == cn[0] and (cn[0] != "box" or iou(o[1], cn[1]) >= 0.3) for o in others) >= 2
        fl, fs = compute_final(row, agree_label, claude_checked, claude_majority)
        row["final_label"] = format_vote(fl) if fl not in ("", None) else ""
        row["final_source"] = fs
        rows.append(row)

    fieldnames = fields if fields else (CSV_COLS_BASE + ["first"] + voter_names + CSV_COLS_TAIL)
    write_csv_atomic(os.path.join(lab_label, "votes.csv"), rows, fieldnames)
    print(f"merged {len(answers)} claude answers, rewrote {len(rows)} rows")


def final_mode(lab_label=LAB_LABEL):
    existing, _ = read_csv(os.path.join(lab_label, "votes.csv"))
    out = {}
    counts = defaultdict(int)
    for tid, row in existing.items():
        fl = row.get("final_label", "")
        fs = row.get("final_source", "")
        if fl == "box" and row.get("human_box"):  # box drawn in the review UI, 330-px tile coords
            fl = row["human_box"]
        if fl in ("", None):  # not settled yet (waits for Claude or a person): left out of the label set
            fl = "u"
        if fl not in ("", None):
            # parse list form if json
            if fl.startswith("["):
                try:
                    flv = json.loads(fl)
                except Exception:
                    flv = fl
            else:
                flv = fl
            out[str(tid)] = flv
        counts[fs or "empty"] += 1
    json.dump(out, open(os.path.join(lab_label, "lab_decisions_final.json"), "w", encoding="utf-8"))
    print(f"wrote {len(out)} final labels to lab_decisions_final.json")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "tally"
    if mode == "tally":
        tally()
    elif mode == "claude":
        claude_mode()
    elif mode == "final":
        final_mode()
    else:
        sys.exit(f"unknown mode: {mode}")
