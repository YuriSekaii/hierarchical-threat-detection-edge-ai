"""How visible is each lab test knife? One label per labelled knife, by eye, for target-split scoring
(eval_stage1_targets.py): recall should be 90-95 % on knives a person can see clearly; fast swings need precision.

Labels: c = the knife is clearly visible (its shape can be seen), b = blurred by motion (a smear, shape unclear),
h = hidden or barely visible (in the fist, behind the body, too dark or too small), x = no knife at the green box.

  .venv/Scripts/python.exe results/samples/poc/label_stage1_visibility.py sheets   # items.json + numbered 4x3 sheets
  # voters write vis_review/votes/<voter>.json {id: letter}; Claude writes votes/claude.json {id: {"label", "sure"}}
  .venv/Scripts/python.exe results/samples/poc/label_stage1_visibility.py tally    # -> visibility.json, needs_claude.json
  .venv/Scripts/python.exe results/samples/poc/label_stage1_visibility.py sheets claude   # sheets of the disagreements only
  .venv/Scripts/python.exe results/samples/poc/label_stage1_visibility.py review   # Claude-unsure knives -> review_visibility.bat
  .venv/Scripts/python.exe results/samples/poc/label_stage1_visibility.py selftest

Rule (as review_stage1_candidates.py): all voters agree -> that label; any difference -> Claude; Claude unsure -> "?"
(kept out of the clear / blurred groups and listed for the human). Test knives = data/lab_stage1_gt test labels plus
results/stage1_kd/cand_review/test_label_fix.json.
"""
import json, os, shutil, sys

import cv2
import numpy as np

ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(REPO, "results", "stage1_kd", "vis_review")
FIX = os.path.join(REPO, "results", "stage1_kd", "cand_review", "test_label_fix.json")
CELL = 320


def knives():
    os.environ["LAB_FIX"] = FIX
    sys.path.insert(0, HERE)
    import sweep_stage1_rules as S
    out = []
    for r in sorted(S.load_cache("lab"), key=lambda r: r["path"]):
        for j, g in enumerate(r["gt"]):
            out.append(dict(id=len(out), img=os.path.relpath(r["path"], REPO).replace("\\", "/"), gt=j, box=g, w=r["w"], h=r["h"]))
    return out


def sheets(only=None, folder="sheets"):
    if only is None:
        items = knives()
        os.makedirs(OUT, exist_ok=True)
        json.dump(items, open(os.path.join(OUT, "items.json"), "w"), indent=0)
    else:  # `sheets claude`: the knives the voters disagreed on, in their own folder
        items = [it for it in json.load(open(os.path.join(OUT, "items.json"))) if it["id"] in set(only)]
    d = os.path.join(OUT, folder)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    index, img, last = {}, None, None
    for n in range(0, len(items), 12):
        cells = []
        for it in items[n:n + 12]:
            if it["img"] != last:
                img, last = cv2.imread(os.path.join(REPO, it["img"])), it["img"]
            w, h = it["w"], it["h"]
            x1, y1, x2, y2 = it["box"][0] * w, it["box"][1] * h, it["box"][2] * w, it["box"][3] * h
            side = int(min(max(300, 3 * max(x2 - x1, y2 - y1)), w, h))
            x0 = int(min(max((x1 + x2) / 2 - side / 2, 0), w - side))
            y0 = int(min(max((y1 + y2) / 2 - side / 2, 0), h - side))
            s = CELL / side
            cell = cv2.resize(img[y0:y0 + side, x0:x0 + side], (CELL, CELL), interpolation=cv2.INTER_CUBIC)
            cv2.rectangle(cell, (int((x1 - x0) * s) - 4, int((y1 - y0) * s) - 4), (int((x2 - x0) * s) + 4, int((y2 - y0) * s) + 4), (0, 255, 0), 2)
            cv2.rectangle(cell, (0, 0), (72, 30), (0, 0, 0), -1)
            cv2.putText(cell, str(it["id"]), (4, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(cell, (0, 0), (CELL - 1, CELL - 1), (60, 60, 60), 1)
            cells.append(cell)
        cells += [np.zeros((CELL, CELL, 3), np.uint8)] * (12 - len(cells))
        name = f"sheet_{n // 12:04d}.jpg"
        cv2.imwrite(os.path.join(d, name), np.vstack([np.hstack(cells[r:r + 4]) for r in range(0, 12, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 92])
        index[name] = [it["id"] for it in items[n:n + 12]]
    json.dump(index, open(os.path.join(d, "index.json"), "w"), indent=0)
    print(f"{len(items)} knives, {len(index)} sheets -> {d}")


def review():
    """Knives Claude was unsure of -> vis_review/human/{items.json, votes.csv} for review_visibility.bat (label_review_ui.py --vis)."""
    import csv
    items = {it["id"]: it for it in json.load(open(os.path.join(OUT, "items.json")))}
    vd = os.path.join(OUT, "votes")
    claude = json.load(open(os.path.join(vd, "claude.json")))
    voters = sorted(f[:-5] for f in os.listdir(vd) if f.endswith(".json") and f != "claude.json")
    votes = {v: json.load(open(os.path.join(vd, v + ".json"))) for v in voters}
    d = os.path.join(OUT, "human")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "votes.csv")
    old = {r["tile_id"]: r for r in csv.DictReader(open(path, encoding="utf-8"))} if os.path.exists(path) else {}
    out, rows = [], []
    for i, c in sorted(claude.items(), key=lambda v: int(v[0])):
        if c.get("sure"):
            continue
        it = items[int(i)]
        w, h = it["w"], it["h"]
        b = [it["box"][0] * w, it["box"][1] * h, it["box"][2] * w, it["box"][3] * h]
        side = int(min(max(300, 3 * max(b[2] - b[0], b[3] - b[1])), w, h))
        x0 = int(min(max((b[0] + b[2]) / 2 - side / 2, 0), w - side))
        y0 = int(min(max((b[1] + b[3]) / 2 - side / 2, 0), h - side))
        take, f = os.path.splitext(os.path.basename(it["img"]))[0].rsplit("_f", 1)
        out.append(dict(id=it["id"], img=it["img"], split="test", take=take, frame=int(f), x0=x0, y0=y0, crop=side, kind="vis",
                        props=[{"letter": "a", "box": [round(v, 1) for v in b], "src": {}}]))
        o = old.get(i, {})
        rows.append(dict(tile_id=i, img=it["img"], split="test", take=take, frame=int(f), kind="vis",
                         **{v: votes[v].get(i, "") for v in voters}, claude_label=c["label"], claude_conf=0.5,
                         human_label=o.get("human_label", ""), final_label=o.get("final_label", ""),
                         final_source=o.get("final_source", "") or "needs_human", last_update=o.get("last_update", "")))
    json.dump(out, open(os.path.join(d, "items.json"), "w"), indent=0)
    fields = ["tile_id", "img", "split", "take", "frame", "kind"] + voters + ["n_votes", "agree", "flag_reason", "claude_label",
              "claude_conf", "human_label", "human_box", "final_label", "final_source", "last_update"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} knives for the human ({sum(bool(r['human_label']) for r in rows)} answered) -> {d}")


def decide(votes, claude=None):
    if votes and None not in votes and len(set(votes)) == 1:
        return votes[0], "vote"
    if claude is None:
        return None, "needs_claude"
    return (claude["label"], "claude") if claude.get("sure") else ("?", "claude_unsure")


assert decide(["c", "c", "c", "c"]) == ("c", "vote")
assert decide(["c", "b", "c", "c"]) == (None, "needs_claude")
assert decide(["c", None, "c"], {"label": "h", "sure": True}) == ("h", "claude")
assert decide(["c", "b"], {"label": "b", "sure": False}) == ("?", "claude_unsure")


def tally():
    items = json.load(open(os.path.join(OUT, "items.json")))
    vd = os.path.join(OUT, "votes")
    voters = sorted(f[:-5] for f in os.listdir(vd) if f.endswith(".json") and f != "claude.json")
    votes = {v: json.load(open(os.path.join(vd, v + ".json"))) for v in voters}
    claude = json.load(open(os.path.join(vd, "claude.json"))) if os.path.exists(os.path.join(vd, "claude.json")) else {}
    hp = os.path.join(OUT, "human", "votes.csv")
    if os.path.exists(hp):  # answers from review_visibility.bat win; "u" stays unsure
        import csv
        for r in csv.DictReader(open(hp, encoding="utf-8")):
            if r.get("human_label") in ("c", "b", "h", "x"):
                claude[r["tile_id"]] = {"label": r["human_label"], "sure": True}
    vis, need, count = {}, [], {}
    for it in items:
        lab, src = decide([votes[v].get(str(it["id"])) for v in voters], claude.get(str(it["id"])))
        if lab is None:
            need.append(it["id"])
        vis.setdefault(os.path.basename(it["img"]), {})[str(it["gt"])] = lab
        count[(lab, src)] = count.get((lab, src), 0) + 1
    # check_stage1_misses.py (ids = items.json ids): d = second label on an already labelled knife, w = box misses the knife
    byid = {str(it["id"]): it for it in items}
    over = dict(json.load(open(os.path.join(OUT, "miss_check", "result.json")))) if os.path.exists(os.path.join(OUT, "miss_check", "result.json")) else {}
    if os.path.exists(os.path.join(OUT, "miss_check", "duplicates.json")):
        over.update({str(i): "d" for i in json.load(open(os.path.join(OUT, "miss_check", "duplicates.json")))["ids"]})
    for i, lab in over.items():
        if lab:
            it = byid[i]
            old = vis[os.path.basename(it["img"])][str(it["gt"])]
            vis[os.path.basename(it["img"])][str(it["gt"])] = lab
            if old != lab:
                count[(lab, "miss_check")] = count.get((lab, "miss_check"), 0) + 1
    json.dump(vis, open(os.path.join(OUT, "visibility.json"), "w"), indent=0)
    json.dump(need, open(os.path.join(OUT, "needs_claude.json"), "w"))
    for k in sorted(count, key=str):
        print(f"  {k[0]!s:4} {k[1]:13} {count[k]}")
    print(f"voters {voters}; needs Claude: {len(need)}")


if __name__ == "__main__":
    {"sheets": lambda: sheets() if ARGS[1:] != ["claude"] else sheets(json.load(open(os.path.join(OUT, "needs_claude.json"))), "claude_sheets"),
     "tally": tally, "review": review, "selftest": lambda: print("selftest ok")}.get(ARGS[0] if ARGS else "", lambda: print(__doc__))()
