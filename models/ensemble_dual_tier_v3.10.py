"""
models/ensemble_dual_tier_v3.10.py

Dual-Tier Edge & Server Architecture Ensembles (Version 3.10 - Final Milestone).

Consolidates all previous research milestones (v1.00 - v3.00) into an integrated
dual-tier surveillance architecture:

1. Tier 1 (Edge Autonomous Screening):
   - Lightweight coordinate-first ST-GCN + Relative Mahalanobis Distance (v1.08 champion).
   - Fast sub-30ms latency on GPU (37.8 FPS) / 26.3 FPS CPU.
   - Autonomous handling of high-confidence threats and clear normal movements.
   - Conditional escalation for borderline / ambiguous events.

2. Tier 2 (Server High-Density Multi-Model Ensemble):
   - Centralized GPU server processing high-density video streams.
   - Combines 3 complementary spatio-temporal representations:
     a) ST-GCN (v1.08, static biological graph)
     b) CTR-GCN (v2.00, dynamic channel-wise topology refinement)
     c) PoseConv3D (v2.10, volumetric 3D heatmap CNN)
   - Multi-model probability fusion with temperature scaling.
   - Resolves multi-person occlusion, grappling, and eliminates false alarms.

3. DualTierSurveillanceSystem:
   - Orchestrates edge screening, ambiguous event escalation, and server verification.
   - Operates in 'edge_only', 'server_only', or 'hierarchical_dual_tier' modes.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from models.stgcn import STGCNModel
from models.ctrgcn import CTRGCNModel
from models.poseconv3d import PoseConv3DModel
from src.poseconv3d_utils import batch_rasterize_gpu, rasterize_keypoints_to_heatmap

# Import RMD metrics
import importlib.util
spec_rmd = importlib.util.spec_from_file_location(
    "ood_metrics_v1_08",
    os.path.join(REPO_DIR, "src", "ood_metrics_v1.08.py")
)
rmd_mod = importlib.util.module_from_spec(spec_rmd)
spec_rmd.loader.exec_module(rmd_mod)
compute_relative_mahalanobis_distance = rmd_mod.compute_relative_mahalanobis_distance


class Tier1EdgeEngine(nn.Module):
    """
    Tier 1: Autonomous Edge Surveillance Engine.
    Operates lightweight ST-GCN with Relative Mahalanobis Distance (RMD).
    """
    def __init__(self, device=None, weights_path=None, ref_path=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights_path = weights_path or os.path.join(REPO_DIR, "weights", "stgcn_violence_v1.02.pth")
        ref_path = ref_path or os.path.join(REPO_DIR, "weights", "reference_data_rmd_v1.08.pt")

        self.model = STGCNModel(num_classes=1).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
        self.model.eval()

        ref_data = torch.load(ref_path, map_location=self.device, weights_only=False)
        self.register_buffer("mu_0", ref_data["mu_0"].to(self.device))
        self.register_buffer("mu_c", ref_data["mu_c"].to(self.device))
        self.register_buffer("inv_Sigma_0", ref_data["inv_Sigma_0"].to(self.device))
        self.register_buffer("inv_Sigma_c", ref_data["inv_Sigma_c"].to(self.device))
        
        self.optimal_threshold = float(ref_data.get("optimal_threshold", -0.9289))
        self.temperature = 1.0

    @torch.no_grad()
    def forward_score(self, clips):
        """
        Computes raw RMD score and calibrated probability for input skeleton clips.
        Args:
            clips: (B, 2, 50, 17) or (2, 50, 17) tensor on self.device.
        Returns:
            rmd_scores (torch.Tensor), probabilities (torch.Tensor)
        """
        if clips.dim() == 3:
            clips = clips.unsqueeze(0)
        clips = clips.to(self.device)

        _, feats = self.model(clips, return_features=True)
        scores, _, _ = compute_relative_mahalanobis_distance(
            feats, self.mu_0, self.mu_c, self.inv_Sigma_0, self.inv_Sigma_c
        )
        probs = torch.sigmoid((scores - self.optimal_threshold) / self.temperature)
        return scores, probs


class Tier2ServerEnsemble(nn.Module):
    """
    Tier 2: Server-Grade High-Density Multi-Model Ensemble.
    Combines:
    1. ST-GCN (static kinematic graph)
    2. CTR-GCN (dynamic channel topology refinement)
    3. PoseConv3D (volumetric 3D heatmap CNN)
    """
    def __init__(
        self,
        device=None,
        weights_stgcn=None,
        ref_stgcn=None,
        weights_ctrgcn=None,
        ref_ctrgcn=None,
        weights_poseconv3d=None,
        ref_poseconv3d=None,
        weights_ensemble=None
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. ST-GCN Backbone
        weights_stgcn = weights_stgcn or os.path.join(REPO_DIR, "weights", "stgcn_violence_v1.02.pth")
        ref_stgcn = ref_stgcn or os.path.join(REPO_DIR, "weights", "reference_data_rmd_v1.08.pt")
        self.stgcn = STGCNModel(num_classes=1).to(self.device)
        self.stgcn.load_state_dict(torch.load(weights_stgcn, map_location=self.device, weights_only=True))
        self.stgcn.eval()
        d_stgcn = torch.load(ref_stgcn, map_location=self.device, weights_only=False)
        self.register_buffer("stgcn_mu_0", d_stgcn["mu_0"].to(self.device))
        self.register_buffer("stgcn_mu_c", d_stgcn["mu_c"].to(self.device))
        self.register_buffer("stgcn_inv_Sigma_0", d_stgcn["inv_Sigma_0"].to(self.device))
        self.register_buffer("stgcn_inv_Sigma_c", d_stgcn["inv_Sigma_c"].to(self.device))
        self.tau_stgcn = float(d_stgcn.get("optimal_threshold", -0.9289))

        # 2. CTR-GCN Backbone
        weights_ctrgcn = weights_ctrgcn or os.path.join(REPO_DIR, "weights", "ctrgcn_violence_v2.00.pth")
        ref_ctrgcn = ref_ctrgcn or os.path.join(REPO_DIR, "weights", "reference_data_ctrgcn_v2.00.pt")
        self.ctrgcn = CTRGCNModel(num_classes=1).to(self.device)
        self.ctrgcn.load_state_dict(torch.load(weights_ctrgcn, map_location=self.device, weights_only=True))
        self.ctrgcn.eval()
        d_ctrgcn = torch.load(ref_ctrgcn, map_location=self.device, weights_only=False)
        self.register_buffer("ctrgcn_mu_0", d_ctrgcn["mu_0"].to(self.device))
        self.register_buffer("ctrgcn_mu_c", d_ctrgcn["mu_c"].to(self.device))
        self.register_buffer("ctrgcn_inv_Sigma_0", d_ctrgcn["inv_Sigma_0"].to(self.device))
        self.register_buffer("ctrgcn_inv_Sigma_c", d_ctrgcn["inv_Sigma_c"].to(self.device))
        self.tau_ctrgcn = float(d_ctrgcn.get("optimal_threshold", 3.1498))

        # 3. PoseConv3D Backbone
        weights_poseconv3d = weights_poseconv3d or os.path.join(REPO_DIR, "weights", "poseconv3d_violence_v2.10.pth")
        ref_poseconv3d = ref_poseconv3d or os.path.join(REPO_DIR, "weights", "reference_data_poseconv3d_v2.10.pt")
        self.poseconv3d = PoseConv3DModel(num_classes=1, base_channels=24).to(self.device)
        self.poseconv3d.load_state_dict(torch.load(weights_poseconv3d, map_location=self.device, weights_only=True))
        self.poseconv3d.eval()
        d_pc3d = torch.load(ref_poseconv3d, map_location=self.device, weights_only=False)
        self.register_buffer("pc3d_mu_0", d_pc3d["mu_0"].to(self.device))
        self.register_buffer("pc3d_mu_c", d_pc3d["mu_c"].to(self.device))
        self.register_buffer("pc3d_inv_Sigma_0", d_pc3d["inv_Sigma_0"].to(self.device))
        self.register_buffer("pc3d_inv_Sigma_c", d_pc3d["inv_Sigma_c"].to(self.device))
        self.tau_pc3d = float(d_pc3d.get("optimal_threshold", -21.4694))

        # Ensemble weights: [w_stgcn, w_ctrgcn, w_poseconv3d]
        if weights_ensemble is None:
            self.weights = torch.tensor([0.45, 0.35, 0.20], device=self.device)
        else:
            self.weights = torch.tensor(weights_ensemble, device=self.device, dtype=torch.float32)
            self.weights = self.weights / self.weights.sum()

        self.tau_server = 0.50

    @torch.no_grad()
    def forward_ensemble_probs(self, clips, heatmaps=None):
        """
        Computes individual model probabilities and fused ensemble probability.
        Args:
            clips: (B, 2, 50, 17) tensor on self.device.
            heatmaps: Optional precomputed (B, 17, 50, 56, 56) tensor.
        Returns:
            dict with 'p_stgcn', 'p_ctrgcn', 'p_pc3d', 'p_ensemble', 'scores'
        """
        if clips.dim() == 3:
            clips = clips.unsqueeze(0)
        clips = clips.to(self.device)
        B = clips.shape[0]

        # 1. ST-GCN Score & Prob
        _, f_stgcn = self.stgcn(clips, return_features=True)
        s_stgcn, _, _ = compute_relative_mahalanobis_distance(
            f_stgcn, self.stgcn_mu_0, self.stgcn_mu_c, self.stgcn_inv_Sigma_0, self.stgcn_inv_Sigma_c
        )
        p_stgcn = torch.sigmoid(s_stgcn - self.tau_stgcn)

        # 2. CTR-GCN Score & Prob
        _, f_ctrgcn = self.ctrgcn(clips, return_features=True)
        s_ctrgcn, _, _ = compute_relative_mahalanobis_distance(
            f_ctrgcn, self.ctrgcn_mu_0, self.ctrgcn_mu_c, self.ctrgcn_inv_Sigma_0, self.ctrgcn_inv_Sigma_c
        )
        p_ctrgcn = torch.sigmoid(s_ctrgcn - self.tau_ctrgcn)

        # 3. PoseConv3D Score & Prob
        if heatmaps is None:
            kpts = clips.permute(0, 2, 3, 1)  # (B, 50, 17, 2)
            if self.device.type == "cuda":
                heatmaps = batch_rasterize_gpu(kpts)
            else:
                hm_list = [rasterize_keypoints_to_heatmap(kpts[b], device="cpu") for b in range(B)]
                heatmaps = torch.stack(hm_list, dim=0)
        else:
            heatmaps = heatmaps.to(self.device)

        _, f_pc3d = self.poseconv3d(heatmaps, return_features=True)
        s_pc3d, _, _ = compute_relative_mahalanobis_distance(
            f_pc3d, self.pc3d_mu_0, self.pc3d_mu_c, self.pc3d_inv_Sigma_0, self.pc3d_inv_Sigma_c
        )
        p_pc3d = torch.sigmoid(s_pc3d - self.tau_pc3d)

        # Fused probability
        w = self.weights
        p_ensemble = w[0] * p_stgcn + w[1] * p_ctrgcn + w[2] * p_pc3d

        return {
            "p_stgcn": p_stgcn,
            "p_ctrgcn": p_ctrgcn,
            "p_pc3d": p_pc3d,
            "p_ensemble": p_ensemble,
            "s_stgcn": s_stgcn,
            "s_ctrgcn": s_ctrgcn,
            "s_pc3d": s_pc3d,
        }


class DualTierSurveillanceSystem(nn.Module):
    """
    Unified Hierarchical Dual-Tier Surveillance System (Version 3.10).
    Manages Tier 1 Edge Screening and conditional Tier 2 Server Verification.
    """
    def __init__(
        self,
        device=None,
        mode="hierarchical_dual_tier",
        p_low=0.35,
        p_high=0.75,
        tau_server=0.50,
        weights_ensemble=None
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.p_low = p_low
        self.p_high = p_high
        self.tau_server = tau_server

        self.tier1_edge = Tier1EdgeEngine(device=self.device)
        self.tier2_server = Tier2ServerEnsemble(device=self.device, weights_ensemble=weights_ensemble)
        self.tier2_server.tau_server = self.tau_server

    def set_thresholds(self, p_low=None, p_high=None, tau_server=None):
        if p_low is not None: self.p_low = p_low
        if p_high is not None: self.p_high = p_high
        if tau_server is not None:
            self.tau_server = tau_server
            self.tier2_server.tau_server = tau_server

    @torch.no_grad()
    def evaluate_clip(self, clip, heatmap=None):
        """
        Processes a single or batch of clips through the dual-tier hierarchy.
        Returns dict with decision status:
        - is_threat: bool / int (1 = threat, 0 = benign)
        - tier_dispatched: "edge" or "server"
        - confidence: float
        - edge_prob: float
        - server_prob: float or None
        """
        if clip.dim() == 3:
            clip = clip.unsqueeze(0)

        # 1. Tier 1 Edge Screening
        _, p_edge = self.tier1_edge.forward_score(clip)

        if self.mode == "edge_only":
            is_threat = (p_edge >= 0.50).long()
            return {
                "is_threat": is_threat,
                "tier_dispatched": ["edge"] * clip.shape[0],
                "confidence": p_edge,
                "edge_prob": p_edge,
                "server_prob": None,
            }

        if self.mode == "server_only":
            res_server = self.tier2_server.forward_ensemble_probs(clip, heatmap)
            p_server = res_server["p_ensemble"]
            is_threat = (p_server >= self.tau_server).long()
            return {
                "is_threat": is_threat,
                "tier_dispatched": ["server"] * clip.shape[0],
                "confidence": p_server,
                "edge_prob": p_edge,
                "server_prob": p_server,
            }

        # 3. Hierarchical Dual-Tier Mode
        B = clip.shape[0]
        final_threat = torch.zeros(B, dtype=torch.long, device=self.device)
        dispatched_tier = []
        confidences = []
        server_probs = []

        for b in range(B):
            prob_e = p_edge[b].item()
            if prob_e < self.p_low:
                # Clear Normal -> Discard at edge
                final_threat[b] = 0
                dispatched_tier.append("edge")
                confidences.append(1.0 - prob_e)
                server_probs.append(None)
            elif prob_e >= self.p_high:
                # Decisive Threat -> Alert immediately at edge
                final_threat[b] = 1
                dispatched_tier.append("edge")
                confidences.append(prob_e)
                server_probs.append(None)
            else:
                # Ambiguous Boundary -> Escalate to Tier 2 Server
                clip_b = clip[b : b + 1]
                hm_b = heatmap[b : b + 1] if heatmap is not None else None
                res_s = self.tier2_server.forward_ensemble_probs(clip_b, hm_b)
                prob_s = res_s["p_ensemble"][0].item()
                server_probs.append(prob_s)

                if prob_s >= self.tau_server:
                    final_threat[b] = 1
                    confidences.append(prob_s)
                else:
                    final_threat[b] = 0
                    confidences.append(1.0 - prob_s)
                dispatched_tier.append("server")

        return {
            "is_threat": final_threat,
            "tier_dispatched": dispatched_tier,
            "confidence": confidences,
            "edge_prob": p_edge,
            "server_prob": server_probs,
        }
