"""
models/ensemble_super.py

Unified Multi-Representation Super-Ensemble Architecture (Phase 2).
Supports arbitrary model combinations from 2 models up to all 9 available backbones:
1. ST-GCN Baseline (v1.08)
2. CTR-GCN Baseline (v2.00)
3. CTR-GCN Tier S (channels=[32, 64, 128], 353k)
4. CTR-GCN Tier L (channels=[96, 192, 384], 2.97M)
5. PoseConv3D Tier 3 (598k)
6. PoseConv3D Tier 5 (132k)
7. PoseConv3D Distilled Tier 3 (Method B RKD, 598k)
8. PoseConv3D Baseline (v2.10, 765k)
9. SkateFormer (v2.20, 447k)

Supports:
- Convex Weighted Soft Voting: P = sum(w_i * P_i)
- Hierarchical Precision-Veto Gating: P >= tau AND P_T5 >= theta_veto
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from models.stgcn import STGCNModel
from models.ctrgcn import CTRGCNModel
from models.ctrgcn_scaled import ScaledCTRGCNModel
from models.poseconv3d import PoseConv3DModel
from models.skateformer import SkateFormerModel

import importlib.util
spec_rmd = importlib.util.spec_from_file_location(
    "ood_metrics_v1_08",
    os.path.join(REPO_DIR, "src", "ood_metrics_v1.08.py")
)
rmd_mod = importlib.util.module_from_spec(spec_rmd)
spec_rmd.loader.exec_module(rmd_mod)
compute_relative_mahalanobis_distance = rmd_mod.compute_relative_mahalanobis_distance


MODEL_REGISTRY = {
    "stgcn": {
        "name": "ST-GCN Baseline",
        "family": "gcn",
        "weights": "weights/stgcn_violence_v1.02.pth",
        "ref": "weights/reference_data_rmd_v1.08.pt"
    },
    "ctrgcn": {
        "name": "CTR-GCN Baseline",
        "family": "ctrgcn",
        "weights": "weights/ctrgcn_violence_v2.00.pth",
        "ref": "weights/reference_data_ctrgcn_v2.00.pt"
    },
    "ctrgcn_tier_s": {
        "name": "CTR-GCN Tier S",
        "family": "ctrgcn_scaled",
        "channels": [32, 64, 128],
        "reduction": 4,
        "arm_weight": 5.0,
        "weights": "weights/ctrgcn_tier_s.pth",
        "ref": "weights/reference_data_ctrgcn_tier_s.pt"
    },
    "ctrgcn_tier_l": {
        "name": "CTR-GCN Tier L",
        "family": "ctrgcn_scaled",
        "channels": [96, 192, 384],
        "reduction": 8,
        "arm_weight": 5.0,
        "weights": "weights/ctrgcn_tier_l.pth",
        "ref": "weights/reference_data_ctrgcn_tier_l.pt"
    },
    "pc3d_tier3": {
        "name": "PoseConv3D Tier 3",
        "family": "poseconv3d",
        "base_channels": 12,
        "feature_dim": 256,
        "weights": "weights/poseconv3d_downscale_tier3_598k.pth",
        "ref_parent": "weights/reference_data_poseconv3d_pure_dual_tier.pt",
        "ref_key": "t3_ref"
    },
    "pc3d_tier5": {
        "name": "PoseConv3D Tier 5",
        "family": "poseconv3d",
        "base_channels": 12,
        "feature_dim": 96,
        "weights": "weights/poseconv3d_downscale_tier5_132k.pth",
        "ref_parent": "weights/reference_data_poseconv3d_pure_dual_tier.pt",
        "ref_key": "t5_ref"
    },
    "pc3d_dt3": {
        "name": "PoseConv3D Distilled Tier 3",
        "family": "poseconv3d",
        "base_channels": 12,
        "feature_dim": 256,
        "weights": "weights/poseconv3d_distill_methodB_rkd.pth",
        "ref_parent": "weights/reference_data_poseconv3d_pure_dual_tier.pt",
        "ref_key": "dt3_ref"
    },
    "pc3d_base": {
        "name": "PoseConv3D Baseline",
        "family": "poseconv3d",
        "base_channels": 24,
        "feature_dim": 256,
        "weights": "weights/poseconv3d_violence_v2.10.pth",
        "ref": "weights/reference_data_poseconv3d_v2.10.pt"
    },
    "skateformer": {
        "name": "SkateFormer ViT",
        "family": "skateformer",
        "weights": "weights/skateformer_violence_v2.20.pth",
        "ref": "weights/reference_data_skateformer_v2.20.pt"
    }
}


def load_model_instance(model_key, device):
    """Factory function to load any registered model and its reference statistics."""
    meta = MODEL_REGISTRY[model_key]
    fam = meta["family"]

    w_path = os.path.join(REPO_DIR, meta["weights"])

    if fam == "gcn":
        model = STGCNModel(num_classes=1).to(device)
        model.load_state_dict(torch.load(w_path, map_location=device, weights_only=True))
    elif fam == "ctrgcn":
        model = CTRGCNModel(num_classes=1).to(device)
        model.load_state_dict(torch.load(w_path, map_location=device, weights_only=True))
    elif fam == "ctrgcn_scaled":
        model = ScaledCTRGCNModel(
            channels=meta["channels"],
            reduction=meta["reduction"],
            arm_weight=meta.get("arm_weight", 5.0)
        ).to(device)
        model.load_state_dict(torch.load(w_path, map_location=device, weights_only=True))
    elif fam == "poseconv3d":
        model = PoseConv3DModel(
            num_classes=1,
            base_channels=meta["base_channels"],
            feature_dim=meta.get("feature_dim", 256)
        ).to(device)
        model.load_state_dict(torch.load(w_path, map_location=device, weights_only=True))
    elif fam == "skateformer":
        model = SkateFormerModel(
            num_classes=1, in_channels=2, num_frames=50, num_keypoints=17,
            embed_dim=128, num_blocks=4
        ).to(device)
        model.load_state_dict(torch.load(w_path, map_location=device, weights_only=True))
    else:
        raise ValueError(f"Unknown family {fam}")

    model.eval()

    # Load reference data
    if "ref" in meta:
        ref = torch.load(os.path.join(REPO_DIR, meta["ref"]), map_location=device, weights_only=False)
    elif "ref_parent" in meta:
        parent_ref = torch.load(os.path.join(REPO_DIR, meta["ref_parent"]), map_location=device, weights_only=False)
        ref = parent_ref[meta["ref_key"]]
    else:
        raise ValueError(f"No reference data specified for {model_key}")

    mu_0 = ref["mu_0"].to(device)
    mu_c = ref["mu_c"].to(device)
    inv_Sigma_0 = ref["inv_Sigma_0"].to(device)
    inv_Sigma_c = ref["inv_Sigma_c"].to(device)
    tau = float(ref.get("optimal_threshold", 0.0))

    return {
        "key": model_key,
        "name": meta["name"],
        "model": model,
        "params": sum(p.numel() for p in model.parameters()),
        "mu_0": mu_0,
        "mu_c": mu_c,
        "inv_Sigma_0": inv_Sigma_0,
        "inv_Sigma_c": inv_Sigma_c,
        "tau": tau,
        "requires_heatmap": fam == "poseconv3d"
    }


class SuperEnsembleSystem(nn.Module):
    """
    Evaluates an arbitrary subset of models with tuned voting weights and thresholds.
    """
    def __init__(self, model_keys, weights=None, threshold=0.50, veto_key=None, veto_thresh=None, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_keys = list(model_keys)
        self.threshold = float(threshold)
        self.veto_key = veto_key
        self.veto_thresh = float(veto_thresh) if veto_thresh is not None else None

        self.models_dict = {}
        for k in self.model_keys:
            self.models_dict[k] = load_model_instance(k, self.device)

        if veto_key is not None and veto_key not in self.models_dict:
            self.models_dict[veto_key] = load_model_instance(veto_key, self.device)

        # Normalize voting weights
        if weights is None:
            w = np.ones(len(self.model_keys)) / len(self.model_keys)
        else:
            w = np.array(weights, dtype=np.float32)
            w = w / w.sum()
        self.weights = {k: float(w[i]) for i, k in enumerate(self.model_keys)}

        self.total_params = sum(m["params"] for m in self.models_dict.values())

    @torch.no_grad()
    def compute_model_probabilities(self, coords, heatmaps=None):
        """
        Computes calibrated posterior probabilities for all constituent models.
        """
        probs = {}
        for k, item in self.models_dict.items():
            model = item["model"]
            if item["requires_heatmap"]:
                assert heatmaps is not None, f"Model {k} requires heatmaps"
                inp = heatmaps.to(self.device)
            else:
                inp = coords.to(self.device)

            _, feats = model(inp, return_features=True)
            scores, _, _ = compute_relative_mahalanobis_distance(
                feats, item["mu_0"], item["mu_c"], item["inv_Sigma_0"], item["inv_Sigma_c"]
            )
            # Clip-level score to probability
            p = torch.sigmoid(scores - item["tau"])
            probs[k] = p.max().item() if p.dim() > 0 else p.item()
        return probs

    def predict_from_probabilities(self, probs_dict):
        """
        Computes ensemble voting probability and binary classification decision.
        """
        p_vote = sum(self.weights[k] * probs_dict[k] for k in self.model_keys)

        decision = int(p_vote >= self.threshold)

        # Precision-veto check
        if decision == 1 and self.veto_key is not None:
            if probs_dict[self.veto_key] < self.veto_thresh:
                decision = 0  # Vetoed by high-precision filter

        return decision, p_vote
