import csv, sys, collections as C
d = sys.argv[1]
V = list(csv.DictReader(open(d + "/poc_videos.csv")))
W = list(csv.DictReader(open(d + "/poc_windows.csv")))
print("missed live-style, by rejecting tier of the BEST window (attack clips):")
out = C.Counter()
for v in V:
    if v["cat"] == "OOD" or v["live"] == "1": continue
    p1, p2, p3 = float(v["p1max"]), float(v["p2max"]), float(v["p3max"])
    ws = [w for w in W if w["file"] == v["file"] and w["cat"] == v["cat"] and w["act"] == v["act"]]
    # per-window reason
    r = C.Counter()
    for w in ws:
        a, b, c = float(w["p1"]), float(w["p2"]), float(w["p3"])
        r["T1 discard" if a < 0.20 else "T2 discard" if b < 0.35 else "T3 below 0.55"] += 1
    out[(v["act"], tuple(sorted(r.items())).__str__())] += 0
    print(f'  {v["cat"][:8]:<9}{v["act"]:<9}{v["file"]:<16} suite={v["suite"]} p1max={p1:.2f} p2max={p2:.2f} p3max={p3:.2f}  {dict(r)}')
print()
print("all attack windows: where they end")
r = C.Counter()
for w in W:
    if w["cat"] == "OOD": continue
    a, b, c = float(w["p1"]), float(w["p2"]), float(w["p3"])
    k = "alarm" if w["verdict"] == "1" else "T1 discard" if a < 0.20 else "T2 discard" if b < 0.35 else "T3 below"
    r[(w["act"], k)] += 1
for k in sorted(r): print("  ", k, r[k])
print()
print("if verdict were ST-GCN alone (p3>=0.55), clips caught live-style:")
for act in ("Thrust", "Stab", "Cut_Down", ""):
    vs = [v for v in V if v["act"] == (act or "OOD")]
    print(f'   {act or "OOD":<9} {sum(float(v["p3max"]) >= 0.55 for v in vs)}/{len(vs)}   T1-only(p1>=0.5): {sum(float(v["p1max"]) >= 0.5 for v in vs)}/{len(vs)}   any tier>=0.55: {sum(max(float(v["p1max"]),float(v["p2max"]),float(v["p3max"])) >= 0.55 for v in vs)}/{len(vs)}')
