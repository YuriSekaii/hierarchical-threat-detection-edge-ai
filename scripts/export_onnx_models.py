"""
scripts/export_onnx_models.py

Official Universal ONNX Blueprint Exporter.
Exports all production PyTorch models to cross-platform .onnx format:
1. Stage 1 Handheld Weapon Detector (Distilled YOLO26s Hungarian NMS-Free)
2. Stage 2a Kinematic Pose Tracker (YOLO26s-Pose 17-Keypoint)
3. Stage 2b Tier 1 Volumetric 3D CNN (PoseConv3D Tier 3, 598k)
4. Stage 2b Tier 2 Volumetric 3D CNN (PoseConv3D Distilled T3, 598k)
5. Stage 2b Tier 3 Spatial-Temporal Graph CNN (ST-GCN, 3.01M)
"""

import os
import sys
import torch
import torch.nn as nn
from ultralytics import YOLO

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from models.ensemble_super import load_model_instance


class ModelFeatureWrapper(nn.Module):
    """Wrapper that ensures both classification logits and feature embeddings are exported."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x, return_features=True)


def export_all():
    weights_dir = os.path.join(REPO_DIR, "weights")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("Exporting to ONNX and compiling engines strictly requires an NVIDIA CUDA GPU.")

    print("\n" + "=" * 80)
    print("   EXPORTING UNIVERSAL ONNX BLUEPRINTS FOR HIERARCHICAL THREAT DETECTION")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    # 1. Stage 1 Weapon Detector
    yolo_pt = os.path.join(weights_dir, "yolo_weapon_distilled.pt")
    yolo_onnx = os.path.join(weights_dir, "yolo_weapon_distilled.onnx")
    print(f"\n[1/5] Exporting Stage 1 Weapon Detector ({os.path.basename(yolo_pt)})...")
    model_yolo = YOLO(yolo_pt)
    model_yolo.export(format="onnx", imgsz=640, half=True, dynamic=False, opset=17)
    print(f"      -> Generated {yolo_onnx} ({os.path.getsize(yolo_onnx)/(1024*1024):.2f} MB)")

    # 2. Stage 2a Pose Extractor
    pose_pt = os.path.join(weights_dir, "yolo26s-pose.pt")
    pose_onnx = os.path.join(weights_dir, "yolo26s-pose.onnx")
    print(f"\n[2/5] Exporting Stage 2a Pose Tracker ({os.path.basename(pose_pt)})...")
    model_pose = YOLO(pose_pt)
    model_pose.export(format="onnx", imgsz=640, half=True, dynamic=False, opset=17)
    print(f"      -> Generated {pose_onnx} ({os.path.getsize(pose_onnx)/(1024*1024):.2f} MB)")

    # 3. Stage 2b Tier 1 (PoseConv3D Tier 3)
    pc3d_t3_onnx = os.path.join(weights_dir, "poseconv3d_downscale_tier3.onnx")
    print(f"\n[3/5] Exporting Stage 2b Tier 1 PoseConv3D T3...")
    item_t1 = load_model_instance("pc3d_tier3", device)
    w_t1 = ModelFeatureWrapper(item_t1["model"]).eval()
    dummy_hm = torch.zeros(1, 17, 50, 56, 56, device=device)
    torch.onnx.export(
        w_t1, dummy_hm, pc3d_t3_onnx,
        input_names=["heatmap"], output_names=["logits", "feats"],
        opset_version=17, dynamic_axes=None
    )
    print(f"      -> Generated {pc3d_t3_onnx} ({os.path.getsize(pc3d_t3_onnx)/(1024*1024):.2f} MB)")

    # 4. Stage 2b Tier 2 (PoseConv3D Distilled T3)
    pc3d_dt3_onnx = os.path.join(weights_dir, "poseconv3d_distill_dt3.onnx")
    print(f"\n[4/5] Exporting Stage 2b Tier 2 PoseConv3D Distilled T3...")
    item_t2 = load_model_instance("pc3d_dt3", device)
    w_t2 = ModelFeatureWrapper(item_t2["model"]).eval()
    torch.onnx.export(
        w_t2, dummy_hm, pc3d_dt3_onnx,
        input_names=["heatmap"], output_names=["logits", "feats"],
        opset_version=17, dynamic_axes=None
    )
    print(f"      -> Generated {pc3d_dt3_onnx} ({os.path.getsize(pc3d_dt3_onnx)/(1024*1024):.2f} MB)")

    # 5. Stage 2b Tier 3 (ST-GCN Baseline)
    stgcn_onnx = os.path.join(weights_dir, "stgcn.onnx")
    print(f"\n[5/5] Exporting Stage 2b Tier 3 ST-GCN...")
    item_t3 = load_model_instance("stgcn", device)
    w_t3 = ModelFeatureWrapper(item_t3["model"]).eval()
    dummy_coords = torch.zeros(1, 2, 50, 17, 1, device=device)
    torch.onnx.export(
        w_t3, dummy_coords, stgcn_onnx,
        input_names=["coords"], output_names=["logits", "feats"],
        opset_version=17, dynamic_axes=None
    )
    print(f"      -> Generated {stgcn_onnx} ({os.path.getsize(stgcn_onnx)/(1024*1024):.2f} MB)")

    print("\n" + "=" * 80)
    print("   ALL 5 ONNX BLUEPRINTS EXPORTED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    export_all()
