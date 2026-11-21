"""Are the clear lab test knives the models miss labelled right? By eye, same rule as label_stage1_visibility.py.

Items: knives labelled clear (vis_review/visibility.json) that B (KD from E) misses at conf 0.20 or F (teacher blur/dark)
misses at conf 0.10, plus the knives labelled "no knife" (x). Each cell shows the label box (green) and B's best box
near it at conf >= 0.05 (yellow, with its conf), if any.

Labels: c = label right (clear knife, green box fits it), w = a knife is clearly there but the green box does not fit it,
b = blurred by motion, h = hidden / barely visible, x = no knife,
m = (human only) a knife is there but hidden / barely visible, and the green box misses it.

  .venv/Scripts/python.exe results/samples/poc/check_stage1_misses.py sheets   # miss_check/{items.json, sheets/}
  # voters write miss_check/votes/<voter>.json {id: letter}; Claude writes votes/claude.json {id: {"label", "sure"}}
  .venv/Scripts/python.exe results/samples/poc/check_stage1_misses.py tally    # -> result.json, needs_claude.json
  .venv/Scripts/python.exe results/samples/poc/check_stage1_misses.py review   # Claude-unsure -> review_misses.bat
"""
import contextlib, csv, io, json, os, shutil, sys

import cv2
import numpy as np

ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
VIS_DIR = os.path.join(REPO, "results", "stage1_kd", "vis_review")
OUT = os.path.join(VIS_DIR, "miss_check")
CELL = 320
sys.path.insert(0, HERE)
from label_stage1_visibility import decide


def crop(it):
    w, h = it["w"], it["h"]
    x1, y1, x2, y2 = it["box"][0] * w, it["box"][1] * h, it["box"][2] * w, it["box"][3] * h
    side = int(min(max(300, 3 * max(x2 - x1, y2 - y1)), w, h))
    x0 = int(min(max((x1 + x2) / 2 - side / 2, 0), w - side))
    y0 = int(min(max((y1 + y2) / 2 - side / 2, 0), h - side))
    return x0, y0, side


def collect():
    with contextlib.redirect_stdout(io.StringIO()):
        import eval_stage1_targets as E
    vis_items = {(os.path.basename(it["img"]), it["gt"]): it for it in json.load(open(os.path.join(VIS_DIR, "items.json")))}
    hitB = E.rows[("B KD from E", 0.20)][0]
    hitF = E.rows[("F teacher blur/dark", 0.10)][0]
    students = [n for n in E.MODELS if n not in ("shipped",)]
    out, k = [], 0
    for r in E.recs:
        name = os.path.basename(r["path"])
        for j, g in enumerate(r["gt"]):
            v = E.VIS.get(name, {}).get(str(j))
            if (v == "c" and not (hitB[k] and hitF[k])) or v == "x":
                it = dict(vis_items[(name, j)])
                cx, cy, diag = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2, max(g[2] - g[0], g[3] - g[1])
                near = [b for b in r["det"]["kdgtNL"] if b["conf"] >= 0.05
                        and abs((b["xyxy"][0] + b["xyxy"][2]) / 2 - cx) < 1.5 * diag and abs((b["xyxy"][1] + b["xyxy"][3]) / 2 - cy) < 1.5 * diag]
                best = max(near, key=lambda b: b["conf"], default=None)
                it.update(vis=v, pred=list(best["xyxy"]) + [best["conf"]] if best else None,
                          missed_by=sum(not E.rows[(n, 0.20)][0][k] for n in students))
                out.append(it)
            k += 1
    return out


def sheets():
    # the votes are keyed to this list; a rerun after the fixes would pick other knives
    assert not os.path.exists(os.path.join(OUT, "items.json")), "miss_check/items.json exists; move it away to rebuild"
    items = collect()
    d = os.path.join(OUT, "sheets")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    json.dump(items, open(os.path.join(OUT, "items.json"), "w"), indent=0)
    index, img, last = {}, None, None
    for n in range(0, len(items), 12):
        cells = []
        for it in items[n:n + 12]:
            if it["img"] != last:
                img, last = cv2.imread(os.path.join(REPO, it["img"])), it["img"]
            x0, y0, side = crop(it)
            s, w, h = CELL / side, it["w"], it["h"]
            cell = cv2.resize(img[y0:y0 + side, x0:x0 + side], (CELL, CELL), interpolation=cv2.INTER_CUBIC)

            def rect(b, col, pad):
                p1 = (int((b[0] * w - x0) * s) - pad, int((b[1] * h - y0) * s) - pad)
                cv2.rectangle(cell, p1, (int((b[2] * w - x0) * s) + pad, int((b[3] * h - y0) * s) + pad), col, 2)
                return p1
            if it["pred"]:
                p1 = rect(it["pred"], (0, 255, 255), 0)
                cv2.putText(cell, f"{it['pred'][4]:.2f}", (max(p1[0], 0), max(p1[1] - 4, 44)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            rect(it["box"], (0, 255, 0), 4)
            cv2.rectangle(cell, (0, 0), (72, 30), (0, 0, 0), -1)
            cv2.putText(cell, str(it["id"]), (4, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(cell, (0, 0), (CELL - 1, CELL - 1), (60, 60, 60), 1)
            cells.append(cell)
        cells += [np.zeros((CELL, CELL, 3), np.uint8)] * (12 - len(cells))
        name = f"sheet_{n // 12:04d}.jpg"
        cv2.imwrite(os.path.join(d, name), np.vstack([np.hstack(cells[r:r + 4]) for r in range(0, 12, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 92])
        index[name] = [it["id"] for it in items[n:n + 12]]
    json.dump(index, open(os.path.join(d, "index.json"), "w"), indent=0)
    print(f"{len(items)} knives ({sum(it['vis'] == 'x' for it in items)} labelled no-knife), {len(index)} sheets -> {d}")


def load():
    vd = os.path.join(OUT, "votes")
    voters = sorted(f[:-5] for f in os.listdir(vd) if f.endswith(".json") and f != "claude.json")
    votes = {v: json.load(open(os.path.join(vd, v + ".json"))) for v in voters}
    cp = os.path.join(vd, "claude.json")
    return voters, votes, json.load(open(cp)) if os.path.exists(cp) else {}


def human(path):
    """{id: answer} from a review .bat's votes.csv; only real answers (u = cannot tell is kept as 'u')."""
    if not os.path.exists(path):
        return {}
    return {r["tile_id"]: r["human_label"] for r in csv.DictReader(open(path, encoding="utf-8")) if r.get("human_label")}


def final():
    """{id: (label, source)}. Order: duplicate (one label per knife, duplicates.json) > new human answer > vote / Claude.
    A result that differs from an earlier human visibility answer goes back to the human ('?', human_conflict)."""
    items = json.load(open(os.path.join(OUT, "items.json")))
    voters, votes, claude = load()
    new = human(os.path.join(OUT, "human", "votes.csv"))
    old = human(os.path.join(VIS_DIR, "human", "votes.csv"))
    dp = os.path.join(OUT, "duplicates.json")
    dup = set(map(str, json.load(open(dp))["ids"])) if os.path.exists(dp) else set()
    out = {}
    for it in items:
        i = str(it["id"])
        if i in dup:
            out[i] = ("d", "duplicate")
        elif new.get(i) in ("c", "w", "m", "b", "h", "x"):
            out[i] = (new[i], "human")
        elif new.get(i) == "u":
            out[i] = ("?", "human_unsure")
        else:
            lab, src = decide([votes[v].get(i) for v in voters], claude.get(i))
            if old.get(i) in ("c", "b", "h", "x") and lab == "?":
                lab, src = old[i], "human_earlier"  # Claude unsure: the human's earlier answer stands
            elif old.get(i) in ("c", "b", "h", "x") and lab not in (None, old[i]):
                lab, src = "?", "human_conflict"
            out[i] = (lab, src)
    return items, out


assert decide(["c", "c", "c", "c"]) == ("c", "vote")


def tally():
    items, fin = final()
    res, need, count = {}, [], {}
    for it in items:
        lab, src = fin[str(it["id"])]
        if lab is None:
            need.append(it["id"])
        res[str(it["id"])] = lab
        count[(it["vis"], lab, src)] = count.get((it["vis"], lab, src), 0) + 1
    json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=0)
    json.dump(need, open(os.path.join(OUT, "needs_claude.json"), "w"))
    for k in sorted(count, key=str):
        print(f"  was {k[0]}  now {k[1]!s:4} {k[2]:13} {count[k]}")
    print(f"needs Claude: {len(need)}")


def review():
    """Claude-unsure knives and conflicts with an earlier human answer -> miss_check/human/{items.json, votes.csv}
    for review_misses.bat (label_review_ui.py --vis --misses)."""
    items_l, fin = final()
    items = {it["id"]: it for it in items_l}
    voters, votes, claude = load()
    prev = human(os.path.join(VIS_DIR, "human", "votes.csv"))
    d = os.path.join(OUT, "human")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "votes.csv")
    old = {r["tile_id"]: r for r in csv.DictReader(open(path, encoding="utf-8"))} if os.path.exists(path) else {}
    out, rows = [], []
    for i in sorted(fin, key=int):
        src = fin[i][1]
        if src not in ("claude_unsure", "human_conflict", "human", "human_unsure"):
            continue
        if src in ("human", "human_unsure") and i not in old:
            continue
        c = claude.get(i, {"label": ""})
        it = items[int(i)]
        w, h = it["w"], it["h"]
        x0, y0, side = crop(it)
        take, f = os.path.splitext(os.path.basename(it["img"]))[0].rsplit("_f", 1)
        props = [{"letter": "a", "box": [round(v, 1) for v in (it["box"][0] * w, it["box"][1] * h, it["box"][2] * w, it["box"][3] * h)], "src": {}}]
        out.append(dict(id=it["id"], img=it["img"], split="test", take=take, frame=int(f), x0=x0, y0=y0, crop=side, kind="miss", props=props))
        o = old.get(i, {})
        rows.append(dict(tile_id=i, img=it["img"], split="test", take=take, frame=int(f), kind="miss",
                         **{v: votes[v].get(i, "") for v in voters}, claude_label=c["label"], claude_conf=0.5,
                         flag_reason=f"your earlier answer: {prev[i]}" if src == "human_conflict" else "",
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


if __name__ == "__main__":
    {"sheets": sheets, "tally": tally, "review": review}.get(ARGS[0] if ARGS else "", lambda: print(__doc__))()
