"""
src/inference_hoi_contact_v4_30.py

Version 4.30: Contact-State HOI Transformer Modular Inference Engine
Implements dual-stream Human-Object Interaction (HOI) detection with physical
bipartite contact-state gating.

Key Innovations:
1. Dual-Stream Architecture:
   - Stream 1: High-sensitivity Distilled YOLO26s (conf=0.20) proposes candidate weapon boxes.
   - Stream 2: Kinematic YOLO26s-Pose extracts human skeletons and hand locations (wrists 9 & 10).
2. Physical Bipartite Spatial Interaction Graph:
   - Evaluates normalized spatial proximity between candidate weapon proposals and nearest hands:
     d_norm = ||c_obj - c_wrist||_2 / R_hand
   - Hard-blocks ungrasped artifacts (wood logs, rails, clothing seams, ground shadows) where
     d_norm > R_interact.
3. DINOv2 Contact-State Gating (S_contact in {no-contact, self-contact, object-contact}):
   - Extracts self-supervised ViT patch/CLS embeddings from candidate crops.
   - Evaluates latent manifold affinity against calibrated weapon-contact vs. empty-hand reference banks.
   - Suppresses bare hands, clenched fists, and open palms where S_contact < tau_contact.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel


REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_WEAPON_WEIGHTS = os.path.join(REPO_DIR, "weights", "yolo_weapon_distilled.pt")
DEFAULT_POSE_WEIGHTS = os.path.join(REPO_DIR, "weights", "yolo26s-pose.pt")
DEFAULT_REFERENCE_WEIGHTS = os.path.join(REPO_DIR, "weights", "reference_data_hoi_contact_v4_30.pt")
DEFAULT_DINOV2_MODEL_ID = "facebook/dinov2-small"


class HOIContactDetector:
    """
    Modular detector engine for Contact-State HOI Transformer (Version 4.30).
    Encapsulates Distilled YOLO proposals, YOLO-Pose kinematic hand extraction,
    bipartite spatial graph association, and DINOv2 contact-state classification.
    """

    def __init__(
        self,
        weapon_weights_path: str = DEFAULT_WEAPON_WEIGHTS,
        pose_weights_path: str = DEFAULT_POSE_WEIGHTS,
        reference_data_path: str = DEFAULT_REFERENCE_WEIGHTS,
        dinov2_model_id: str = DEFAULT_DINOV2_MODEL_ID,
        device: Optional[str] = None,
        weapon_conf_thresh: float = 0.20,
        pose_conf_thresh: float = 0.25,
        wrist_conf_thresh: float = 0.20,
        interaction_radius: float = 2.2,
        contact_thresh: float = 0.50,
        high_conf_bypass: float = 0.70,
        top_k_sim: int = 3,
        use_amp: bool = True,
    ):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.weapon_conf_thresh = weapon_conf_thresh
        self.pose_conf_thresh = pose_conf_thresh
        self.wrist_conf_thresh = wrist_conf_thresh
        self.interaction_radius = interaction_radius
        self.contact_thresh = contact_thresh
        self.high_conf_bypass = high_conf_bypass
        self.top_k_sim = top_k_sim
        self.use_amp = use_amp and ("cuda" in self.device)

        print(f"[HOI Contact v4.30] Initializing Dual-Stream Pipeline on {self.device}...")

        # 1. Load Stream 1: Weapon Detector
        if not os.path.exists(weapon_weights_path):
            raise FileNotFoundError(f"Weapon weights not found: {weapon_weights_path}")
        self.weapon_model = YOLO(weapon_weights_path)
        print(f"  Stream 1: Loaded Weapon Detector from {os.path.basename(weapon_weights_path)}")

        # 2. Load Stream 2: Pose Estimator
        if not os.path.exists(pose_weights_path):
            raise FileNotFoundError(f"Pose weights not found: {pose_weights_path}")
        self.pose_model = YOLO(pose_weights_path)
        print(f"  Stream 2: Loaded Pose Estimator from {os.path.basename(pose_weights_path)}")

        # 3. Load DINOv2 Feature Extractor
        print(f"  Contact Stream: Loading DINOv2 ({dinov2_model_id})...")
        self.processor = AutoImageProcessor.from_pretrained(dinov2_model_id)
        self.dino_model = AutoModel.from_pretrained(dinov2_model_id).to(self.device).eval()

        # 4. Load Calibrated Reference Feature Bank
        if os.path.exists(reference_data_path):
            print(f"  Loading Calibrated Reference Bank from {os.path.basename(reference_data_path)}...")
            ref_dict = torch.load(reference_data_path, map_location=self.device, weights_only=False)
            self.weapon_ref_features = ref_dict["weapon_features"].to(self.device)
            self.hand_ref_features = ref_dict["hand_features"].to(self.device)
            self.mean_weapon_proto = ref_dict["mean_weapon_prototype"].to(self.device)
            self.mean_hand_proto = ref_dict["mean_hand_prototype"].to(self.device)
            print(f"  Reference Bank loaded: {self.weapon_ref_features.shape[0]} weapon crops, {self.hand_ref_features.shape[0]} hand crops.")
        else:
            print(f"  [WARNING] Reference bank not found at {reference_data_path}. Gating disabled.")
            self.weapon_ref_features = None
            self.hand_ref_features = None
            self.mean_weapon_proto = None
            self.mean_hand_proto = None

        vram_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2) if "cuda" in self.device else 0
        print(f"[HOI Contact v4.30] Initialization complete. Active VRAM footprint: {vram_mb:.2f} MB")

    def _extract_hands(self, pose_result, img_w: int, img_h: int) -> List[Dict]:
        """
        Extracts hand locations and dynamic interaction radii from YOLO pose keypoints.
        COCO keypoints:
        Left: shoulder 5, elbow 7, wrist 9
        Right: shoulder 6, elbow 8, wrist 10
        """
        hands = []
        if pose_result.keypoints is None or len(pose_result.keypoints) == 0:
            return hands

        kpts = pose_result.keypoints.data.cpu().numpy()
        for p_idx in range(len(kpts)):
            person_kpt = kpts[p_idx]
            for side, wrist_idx, elbow_idx in [("left", 9, 7), ("right", 10, 8)]:
                wx, wy, wc = person_kpt[wrist_idx]
                ex, ey, ec = person_kpt[elbow_idx]

                if wc >= self.wrist_conf_thresh:
                    if ec >= self.wrist_conf_thresh:
                        forearm_len = np.linalg.norm(np.array([wx, wy]) - np.array([ex, ey]))
                        radius = forearm_len * 0.50
                    else:
                        radius = 45.0
                    radius = float(max(25.0, min(radius, 150.0)))

                    x1 = max(0, int(wx - radius))
                    y1 = max(0, int(wy - radius))
                    x2 = min(img_w, int(wx + radius))
                    y2 = min(img_h, int(wy + radius))

                    hands.append({
                        "person_id": p_idx,
                        "side": side,
                        "wrist": (float(wx), float(wy)),
                        "conf": float(wc),
                        "radius": radius,
                        "box": [x1, y1, x2, y2]
                    })
        return hands

    def _evaluate_contact_state(self, crop_bgr: np.ndarray) -> Tuple[float, str]:
        """
        Extracts DINOv2 self-supervised patch embeddings and computes contact score
        relative to the calibrated weapon-contact vs. empty-hand manifold.
        """
        if self.weapon_ref_features is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
            return 0.50, "indeterminate"

        pil_crop = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=pil_crop, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            if self.use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.dino_model(**inputs)
            else:
                outputs = self.dino_model(**inputs)

            feat = F.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=-1)

            # Cosine similarities to weapon and hand banks
            sim_weapon = (feat @ self.weapon_ref_features.T).squeeze(0) # [Nw]
            top_sim_weapon = sim_weapon.topk(k=min(self.top_k_sim, len(sim_weapon)))[0].mean().item()

            sim_hand = (feat @ self.hand_ref_features.T).squeeze(0) # [Nh]
            top_sim_hand = sim_hand.topk(k=min(self.top_k_sim, len(sim_hand)))[0].mean().item()

            # Normalized contact affinity score
            contact_score = top_sim_weapon / (top_sim_weapon + top_sim_hand + 1e-6)

            if contact_score >= self.contact_thresh:
                contact_state = "object-contact"
            else:
                contact_state = "self-contact"

        return float(contact_score), contact_state

    def predict(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        weapon_conf_thresh: Optional[float] = None,
        interaction_radius: Optional[float] = None,
        contact_thresh: Optional[float] = None,
    ) -> Dict:
        """
        Executes end-to-end dual-stream inference on a single surveillance frame.
        """
        t0 = time.perf_counter()

        # Resolve thresholds
        w_conf = weapon_conf_thresh if weapon_conf_thresh is not None else self.weapon_conf_thresh
        r_interact = interaction_radius if interaction_radius is not None else self.interaction_radius
        c_thresh = contact_thresh if contact_thresh is not None else self.contact_thresh

        # Load image
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise FileNotFoundError(f"Image not found at: {image_input}")
        elif isinstance(image_input, Image.Image):
            img_bgr = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        h_img, w_img = img_bgr.shape[:2]

        # -------------------------------------------------------------
        # Stream 1: Candidate Weapon Proposals (Distilled YOLO)
        # -------------------------------------------------------------
        weapon_results = self.weapon_model(img_bgr, conf=w_conf, verbose=False)[0]
        raw_proposals = []
        for b in weapon_results.boxes:
            box_coords = b.xyxy[0].cpu().numpy().tolist() # [x1, y1, x2, y2]
            conf = float(b.conf[0])
            raw_proposals.append((box_coords, conf))

        # -------------------------------------------------------------
        # Stream 2: Kinematic Pose & Hand Localization
        # -------------------------------------------------------------
        pose_results = self.pose_model(img_bgr, conf=self.pose_conf_thresh, verbose=False)[0]
        detected_hands = self._extract_hands(pose_results, w_img, h_img)

        # -------------------------------------------------------------
        # Bipartite Graph Association & DINOv2 Contact Gating
        # -------------------------------------------------------------
        validated_boxes = []
        validated_scores = []
        validated_contact_states = []
        validated_distances = []
        suppressed_details = []

        for p_box, p_conf in raw_proposals:
            c_obj = np.array([(p_box[0] + p_box[2]) / 2.0, (p_box[1] + p_box[3]) / 2.0])

            # Compute distance to nearest hand
            min_norm_dist = float("inf")
            nearest_hand = None
            for h in detected_hands:
                c_hand = np.array(h["wrist"])
                d_norm = np.linalg.norm(c_obj - c_hand) / (h["radius"] + 1e-6)
                if d_norm < min_norm_dist:
                    min_norm_dist = d_norm
                    nearest_hand = h

            # High confidence bypass (e.g. >= 0.70)
            if p_conf >= self.high_conf_bypass:
                validated_boxes.append(p_box)
                validated_scores.append(p_conf)
                validated_contact_states.append("object-contact (high-conf-bypass)")
                validated_distances.append(min_norm_dist if min_norm_dist != float("inf") else -1.0)
                continue

            # Check spatial interaction radius
            if min_norm_dist > r_interact:
                # Ungrasped artifact (clothing seam, wood log, floor shadow)
                suppressed_details.append({
                    "box": p_box,
                    "conf": p_conf,
                    "reason": "ungrasped_spatial_rejection",
                    "norm_dist": min_norm_dist
                })
                continue

            # Candidate is near hand -> Evaluate DINOv2 Contact State
            x1, y1 = max(0, int(p_box[0])), max(0, int(p_box[1]))
            x2, y2 = min(w_img, int(p_box[2])), min(h_img, int(p_box[3]))
            crop = img_bgr[y1:y2, x1:x2]

            contact_score, contact_state = self._evaluate_contact_state(crop)

            if contact_score >= c_thresh:
                # Confirmed object contact
                validated_boxes.append(p_box)
                validated_scores.append(p_conf)
                validated_contact_states.append(contact_state)
                validated_distances.append(min_norm_dist)
            else:
                # Empty bare hand or clenched fist
                suppressed_details.append({
                    "box": p_box,
                    "conf": p_conf,
                    "reason": "empty_hand_contact_rejection",
                    "contact_score": contact_score,
                    "norm_dist": min_norm_dist
                })

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "boxes": validated_boxes,
            "scores": validated_scores,
            "contact_states": validated_contact_states,
            "hand_distances": validated_distances,
            "raw_proposal_count": len(raw_proposals),
            "detected_hand_count": len(detected_hands),
            "suppressed_proposals": suppressed_details,
            "latency_ms": latency_ms,
            "image_size": (w_img, h_img)
        }
