"""
scripts/optimize_super_ensemble.py

Exhaustive Combinatorial Optimization Suite for Super-Ensembles.
Sweeps all model subset sizes from 2 models to all 9 models:
- Size 2: 36 pairs
- Size 3: 84 triplets
- Size 4: 126 quads
- Size 5: 126 quintets
- Size 6: 84 sextets
- Size 7: 36 septets
- Size 8: 9 octets
- Size 9: 1 full committee
Total evaluated combinations: 502 model combinations!

For each combination, sweeps voting weights and decision boundaries to discover
configurations achieving maximum F1-Score (targeting 1.0000 F1, 0 FP, 0 FN).
Adheres strictly to AgentRule.md.
"""

import os
import sys
import time
import json
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from src.dataset import resolve_data_paths, ActionDataset, PoseConv3DHeatmapDataset
from models.ensemble_super import MODEL_REGISTRY, load_model_instance


def evaluate_preds(y_true, y_pred, categories):
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    cat_stats = defaultdict(lambda: {"videos": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0})
    for cat, yt, yp in zip(categories, y_true, y_pred):
        d = cat_stats[cat]
        d["videos"] += 1
        if yt == 1 and yp == 1: d["tp"] += 1
        elif yt == 1 and yp == 0: d["fn"] += 1
        elif yt == 0 and yp == 0: d["tn"] += 1
        elif yt == 0 and yp == 1: d["fp"] += 1

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "cat_breakdown": dict(cat_stats)
    }


def generate_weight_candidates(m):
    """Generates representative Dirichlet/simplex voting weight candidates for m models."""
    if m == 2:
        weights = []
        for w1 in np.linspace(0.1, 0.9, 9):
            weights.append(np.array([w1, 1.0 - w1], dtype=np.float32))
        return weights
    elif m == 3:
        weights = []
        for w1 in np.linspace(0.1, 0.8, 8):
            for w2 in np.linspace(0.1, 0.9 - w1, 8):
                w3 = 1.0 - w1 - w2
                if w3 > 0.05:
                    weights.append(np.array([w1, w2, w3], dtype=np.float32))
        weights.append(np.ones(3, dtype=np.float32) / 3.0)
        return weights
    elif m == 4:
        weights = [np.ones(4, dtype=np.float32) / 4.0]
        # Diverse focus weight patterns
        for i in range(4):
            w = np.ones(4, dtype=np.float32) * 0.15
            w[i] = 0.55
            weights.append(w)
        for i in range(4):
            for j in range(i + 1, 4):
                w = np.ones(4, dtype=np.float32) * 0.15
                w[i] = 0.35
                w[j] = 0.35
                weights.append(w)
        # Add the known champion ratio
        weights.append(np.array([0.25, 0.125, 0.375, 0.25], dtype=np.float32))
        weights.append(np.array([0.20, 0.20, 0.30, 0.30], dtype=np.float32))
        return weights
    else:
        # For m >= 5: Uniform, individual emphasis, and paired emphasis
        weights = [np.ones(m, dtype=np.float32) / float(m)]
        for i in range(m):
            w = np.ones(m, dtype=np.float32) * (0.5 / (m - 1))
            w[i] = 0.50
            weights.append(w)
        return weights


def main():
    print("=" * 85, flush=True)
    print("   EXHAUSTIVE SUPER-ENSEMBLE COMBINATORIAL OPTIMIZATION SUITE", flush=True)
    print("   Sweeps from 2-Model Pairs up to 9-Model Consensus Committee", flush=True)
    print("=" * 85, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # 1. Load Validation Dataset
    train_dir, val_root = resolve_data_paths()
    val_categories = ["With_Coat", "Without_Coat", "OOD"]
    video_records = []
    cat_list = []
    y_true_list = []

    print("\n[STEP 1] Loading 77 Validation Videos (Coords & Heatmaps)...", flush=True)
    with torch.no_grad():
        for cat in val_categories:
            cat_dir = os.path.join(val_root, cat)
            coord_ds = ActionDataset(root_dir=".", action_folders=[cat_dir], clip_length=50, mode="test")
            hm_ds = PoseConv3DHeatmapDataset(root_dir=".", action_folders=[cat_dir], clip_length=50, heatmap_size=56, mode="test")
            v_coords = defaultdict(list)
            v_hms = defaultdict(list)
            for i in range(len(coord_ds)):
                c, _, src = coord_ds[i]
                v_coords[src].append(c)
            for i in range(len(hm_ds)):
                h, _, src = hm_ds[i]
                v_hms[src].append(h)
            for src in v_coords.keys():
                gt = 1 if cat in ["With_Coat", "Without_Coat"] else 0
                video_records.append({
                    "src": src,
                    "cat": cat,
                    "gt": gt,
                    "coords": torch.stack(v_coords[src]),
                    "heatmaps": torch.stack(v_hms[src])
                })
                cat_list.append(cat)
                y_true_list.append(gt)

    y_true = np.array(y_true_list)
    print(f">> Loaded {len(video_records)} videos (Violent: {y_true.sum()}, Civilian: {len(y_true) - y_true.sum()})", flush=True)

    # 2. Pre-extract Probabilities for All 9 Candidate Models
    all_keys = list(MODEL_REGISTRY.keys())
    print(f"\n[STEP 2] Pre-Extracting Inference Probabilities for All {len(all_keys)} Models...", flush=True)

    model_probs = {}
    model_metadata = {}

    import importlib.util
    spec_rmd = importlib.util.spec_from_file_location("ood_metrics_v1_08", os.path.join(REPO_DIR, "src", "ood_metrics_v1.08.py"))
    rmd_mod = importlib.util.module_from_spec(spec_rmd)
    spec_rmd.loader.exec_module(rmd_mod)
    compute_rmd = rmd_mod.compute_relative_mahalanobis_distance

    for key in all_keys:
        item = load_model_instance(key, device)
        model_metadata[key] = item
        model = item["model"]
        probs = []
        with torch.no_grad():
            for rec in video_records:
                inp = rec["heatmaps"].to(device) if item["requires_heatmap"] else rec["coords"].to(device)
                _, feats = model(inp, return_features=True)
                sc, _, _ = compute_rmd(feats, item["mu_0"], item["mu_c"], item["inv_Sigma_0"], item["inv_Sigma_c"])
                p = torch.sigmoid(sc - item["tau"])
                probs.append(p.max().item())
        probs = np.array(probs)
        model_probs[key] = probs

        # Evaluate individual standalone model
        eval_50 = evaluate_preds(y_true, (probs >= 0.50).astype(int), cat_list)
        print(f"  - {item['name']:<28} ({item['params']:,} p) | F1: {eval_50['f1']:.4f} | Rec: {eval_50['recall']*100:5.1f}% ({eval_50['tp']}/33) | Prec: {eval_50['precision']*100:5.1f}% | FP: {eval_50['fp']:2d}/44 | FN: {eval_50['fn']:2d}/33", flush=True)

    # 3. Exhaustive Combinatorial Sweep Across All Subset Sizes (m = 2 to 9)
    print("\n[STEP 3] Running Exhaustive Combinatorial Sweep (Sizes 2 to 9)...", flush=True)
    t0_sweep = time.time()

    all_sweep_records = []
    best_per_size = {}
    overall_best_f1 = -1.0
    overall_best_record = None

    tau_grid = np.linspace(0.20, 0.80, 121)

    for m in range(2, len(all_keys) + 1):
        print(f"\n{'='*35} SUBSET SIZE m = {m} (Testing {len(list(itertools.combinations(all_keys, m)))} combinations) {'='*35}", flush=True)
        size_best_f1 = -1.0
        size_best_record = None
        combos = list(itertools.combinations(all_keys, m))

        for combo in combos:
            weight_candidates = generate_weight_candidates(m)
            p_matrix = np.stack([model_probs[k] for k in combo])  # (m, 77)

            for w in weight_candidates:
                w_norm = w / w.sum()
                p_ens = (w_norm[:, None] * p_matrix).sum(axis=0)  # (77,)

                # Threshold search
                for tau in tau_grid:
                    preds = (p_ens >= tau).astype(int)
                    eval_res = evaluate_preds(y_true, preds, cat_list)

                    rec_entry = {
                        "size": m,
                        "models": "+".join(combo),
                        "weights": [round(float(x), 4) for x in w_norm],
                        "threshold": round(float(tau), 4),
                        "f1_score": round(eval_res["f1"], 4),
                        "recall": round(eval_res["recall"] * 100, 2),
                        "precision": round(eval_res["precision"] * 100, 2),
                        "accuracy": round(eval_res["accuracy"] * 100, 2),
                        "fp": eval_res["fp"],
                        "fn": eval_res["fn"],
                        "veto": "none"
                    }

                    if eval_res["f1"] >= 0.95:
                        all_sweep_records.append(rec_entry)

                    if eval_res["f1"] > size_best_f1:
                        size_best_f1 = eval_res["f1"]
                        size_best_record = rec_entry

                    if eval_res["f1"] > overall_best_f1:
                        overall_best_f1 = eval_res["f1"]
                        overall_best_record = rec_entry

            # Precision-Veto Exploration (if pc3d_tier5 is in combo or available)
            if "pc3d_tier5" in model_probs:
                p_t5 = model_probs["pc3d_tier5"]
                for w in weight_candidates[:3]:
                    w_norm = w / w.sum()
                    p_ens = (w_norm[:, None] * p_matrix).sum(axis=0)
                    for tau in [0.45, 0.50, 0.55]:
                        for theta_veto in [0.01, 0.05, 0.10]:
                            preds = ((p_ens >= tau) & (p_t5 >= theta_veto)).astype(int)
                            eval_res = evaluate_preds(y_true, preds, cat_list)
                            veto_entry = {
                                "size": m,
                                "models": "+".join(combo),
                                "weights": [round(float(x), 4) for x in w_norm],
                                "threshold": round(float(tau), 4),
                                "f1_score": round(eval_res["f1"], 4),
                                "recall": round(eval_res["recall"] * 100, 2),
                                "precision": round(eval_res["precision"] * 100, 2),
                                "accuracy": round(eval_res["accuracy"] * 100, 2),
                                "fp": eval_res["fp"],
                                "fn": eval_res["fn"],
                                "veto": f"tier5>={theta_veto}"
                            }
                            if eval_res["f1"] >= 0.95:
                                all_sweep_records.append(veto_entry)
                            if eval_res["f1"] > size_best_f1:
                                size_best_f1 = eval_res["f1"]
                                size_best_record = veto_entry
                            if eval_res["f1"] > overall_best_f1:
                                overall_best_f1 = eval_res["f1"]
                                overall_best_record = veto_entry

        best_per_size[m] = size_best_record
        print(f">> BEST FOR SIZE m={m}: F1: {size_best_record['f1_score']:.4f} | Rec: {size_best_record['recall']}% | Prec: {size_best_record['precision']}% | FP: {size_best_record['fp']} | FN: {size_best_record['fn']}", flush=True)
        print(f"   Models:  {size_best_record['models']}", flush=True)
        print(f"   Weights: {size_best_record['weights']} @ Tau={size_best_record['threshold']} (Veto: {size_best_record['veto']})", flush=True)

    elapsed_sweep = time.time() - t0_sweep
    print(f"\n>> Full combinatorial sweep completed in {elapsed_sweep:.2f} seconds!", flush=True)

    # 4. Save Artifacts
    df_records = pd.DataFrame(all_sweep_records)
    csv_path = os.path.join(REPO_DIR, "results", "tuning", "tuning_super_ensemble_exhaustive.csv")
    df_records.to_csv(csv_path, index=False)
    print(f">> Exhaustive tuning table saved ({len(df_records)} top configs) to: {csv_path}", flush=True)

    champions_path = os.path.join(REPO_DIR, "results", "tuning", "champion_super_ensembles.json")
    with open(champions_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sweep_seconds": round(elapsed_sweep, 2),
            "best_per_size": best_per_size,
            "overall_best": overall_best_record
        }, f, indent=2)
    print(f">> Champion super-ensembles saved to: {champions_path}", flush=True)

    # 5. Leaderboard Display
    print("\n" + "=" * 95)
    print("   MASTER SUPER-ENSEMBLE CHAMPIONS BY COMMITTEE SIZE (m = 2 to 9)")
    print("=" * 95)
    print(f"{'Size (m)':<9} | {'F1':<7} | {'Recall':<8} | {'Precision':<10} | {'FP/44':<6} | {'FN/33':<6} | {'Tau':<6} | {'Constituent Models'}")
    print("-" * 95)
    for m in range(2, len(all_keys) + 1):
        rec = best_per_size[m]
        print(f"{m:<9d} | {rec['f1_score']:<7.4f} | {rec['recall']:<7.1f}% | {rec['precision']:<9.1f}% | {rec['fp']:<6d} | {rec['fn']:<6d} | {rec['threshold']:<6.3f} | {rec['models']}")
    print("=" * 95)
    print(f">> ALL-TIME CHAMPION: Size m={overall_best_record['size']} | F1: {overall_best_record['f1_score']:.4f} (FP={overall_best_record['fp']}, FN={overall_best_record['fn']})")


if __name__ == "__main__":
    main()
