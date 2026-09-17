import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import os

def create_pipeline_diagram(output_path='assets/inference_pipeline.png', dpi=200):
    output_path = os.path.normpath(os.path.abspath(output_path))
    fig = plt.figure(figsize=(19.6, 12.8), dpi=dpi, facecolor='#ffffff')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 148.0)
    ax.set_ylim(0, 104.0)
    ax.axis('off')
    
    # Palette definition
    C_OUTSIDE_BG = '#f8fafc'       # Crisp modern slate canvas
    C_STAGE1_BG = '#f0fdf4'        # Soft emerald tint (Stage 1)
    C_STAGE1_BORDER = '#15803d'    # Emerald border
    C_STAGE2A_BG = '#eff6ff'       # Soft sky blue tint (Stage 2a)
    C_STAGE2A_BORDER = '#1d4ed8'   # Blue border
    C_STAGE2B_BG = '#faf5ff'       # Soft purple tint (Stage 2b)
    C_STAGE2B_BORDER = '#7e22ce'   # Purple border
    
    C_BOX_BG = '#ffffff'           # Process box fill
    C_BOX_BORDER = '#334155'       # Dark slate border
    C_DIAMOND_BG = '#ffffff'       # Decision diamond fill
    C_DIAMOND_BORDER = '#1e293b'   # Decision diamond border
    C_TEXT = '#0f172a'             # Primary text
    C_NOTE_TEXT = '#475569'        # Note/annotation text
    C_ARROW = '#1e293b'            # Arrow lines
    
    # Highlight styling
    C_VIOLENCE_BG = '#fee2e2'      # Soft red fill
    C_VIOLENCE_BORDER = '#b91c1c'  # Crimson border
    C_VIOLENCE_TEXT = '#991b1b'    # Crimson text
    
    C_DISMISS_BG = '#f1f5f9'       # Neutral slate fill
    C_DISMISS_BORDER = '#64748b'   # Muted border
    C_DISMISS_TEXT = '#334155'     # Muted text
    
    C_CHAMP_BG = '#fef3c7'         # Amber champion badge fill
    C_CHAMP_BORDER = '#d97706'     # Amber border
    C_CHAMP_TEXT = '#92400e'       # Amber text

    # Base canvas
    ax.add_patch(Rectangle((0, 0), 148.0, 104.0, facecolor=C_OUTSIDE_BG, edgecolor='none', zorder=0))

    def draw_box(x, y, w, h, title_text, box_type='standard', font_sz=10.5, subtext=None):
        if box_type == 'violence':
            fc, ec, tc, lw = C_VIOLENCE_BG, C_VIOLENCE_BORDER, C_VIOLENCE_TEXT, 2.4
        elif box_type == 'dismiss':
            fc, ec, tc, lw = C_DISMISS_BG, C_DISMISS_BORDER, C_DISMISS_TEXT, 1.6
        elif box_type == 'champ':
            fc, ec, tc, lw = C_CHAMP_BG, C_CHAMP_BORDER, C_CHAMP_TEXT, 2.0
        else:
            fc, ec, tc, lw = C_BOX_BG, C_BOX_BORDER, C_TEXT, 1.8
            
        rect = Rectangle((x - w/2, y - h/2), w, h, 
                         facecolor=fc, edgecolor=ec, linewidth=lw, 
                         zorder=2)
        ax.add_patch(rect)
        
        if subtext:
            ax.text(x, y + 1.2, title_text, ha='center', va='center', 
                    fontsize=font_sz, fontweight='bold', family='sans-serif',
                    color=tc, linespacing=1.2, zorder=3)
            ax.text(x, y - h/2 + 1.4, subtext, ha='center', va='center',
                    fontsize=8.6, family='sans-serif', color=C_NOTE_TEXT, zorder=3)
        else:
            ax.text(x, y, title_text, ha='center', va='center', 
                    fontsize=font_sz, fontweight='bold', family='sans-serif',
                    color=tc, linespacing=1.25, zorder=3)
        return (x, y, w, h)

    def draw_diamond(x, y, w, h, text, font_sz=9.8):
        vertices = [(x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)]
        poly = Polygon(vertices, closed=True, facecolor=C_DIAMOND_BG, 
                       edgecolor=C_DIAMOND_BORDER, linewidth=1.8, zorder=2)
        ax.add_patch(poly)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=font_sz, fontweight='bold', family='sans-serif',
                color=C_TEXT, linespacing=1.2, zorder=3)

    def draw_arrow(x1, y1, x2, y2, zorder=4, color=C_ARROW, lw=1.8):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='-|>',
                                mutation_scale=14,
                                linewidth=lw,
                                color=color,
                                zorder=zorder)
        ax.add_patch(arrow)

    def draw_polyline_arrow(points, zorder=4, color=C_ARROW, lw=1.8):
        for i in range(len(points) - 2):
            ax.add_line(Line2D([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]],
                               color=color, linewidth=lw, zorder=zorder))
        p_prev = points[-2]
        p_last = points[-1]
        draw_arrow(p_prev[0], p_prev[1], p_last[0], p_last[1], zorder=zorder, color=color, lw=lw)

    # Master Title Banner
    ax.text(74.0, 101.5, "Hierarchical Edge-AI Surveillance: End-to-End Threat & Violence Detection Cascade",
            ha='center', va='center', fontsize=15.5, fontweight='heavy', family='sans-serif', color='#0f172a')
    ax.text(74.0, 99.2, "Production Architecture: Distilled NMS-Free YOLO26s  →  Torso-Invariant Tracking  →  Progressive Multi-Tier Staircase Cascade (v5.10)",
            ha='center', va='center', fontsize=10.5, family='sans-serif', color='#334155')

    # =========================================================================
    # STAGE 1: ALWAYS-ON WEAPON SCREENING
    # =========================================================================
    s1_rect = Rectangle((2.0, 69.0), 144.0, 28.5,
                        facecolor=C_STAGE1_BG, edgecolor=C_STAGE1_BORDER,
                        linestyle='--', linewidth=1.8, zorder=1)
    ax.add_patch(s1_rect)
    ax.text(74.0, 95.5, "STAGE 1: Continuous Lightweight Threat Screening (Always-On Ingestion & Gating Loop)",
            ha='center', va='center', fontsize=13.0, fontweight='bold', family='sans-serif', color='#14532d', zorder=3)
    ax.text(4.0, 71.0, "Subsamples 1 in 10 frames (3 FPS) • Cuts idle GPU workloads by ~90% • NMS-Free Hungarian matching cuts seam false alarms by >50%",
            ha='left', va='center', fontsize=9.2, family='sans-serif', color='#166534', zorder=3)

    y_s1 = 82.5
    h_s1 = 8.8

    # 1. Live Video Stream
    draw_box(12.5, y_s1, 17.0, h_s1, "Live Surveillance\nStream (30 FPS)", subtext="RTSP / USB Camera")
    draw_arrow(12.5 + 17.0/2, y_s1, 31.0 - 18.0/2, y_s1)

    # 2. TurboJPEG Circular Buffer
    draw_box(31.0, y_s1, 18.0, h_s1, "TurboJPEG SIMD\nCircular Buffer", subtext="47.3 MB (1,200 f, -95.5% RAM)")
    draw_arrow(31.0 + 18.0/2, y_s1, 49.5 - 16.0/2, y_s1)

    # 3. 3 FPS Subsampling
    draw_box(49.5, y_s1, 16.0, h_s1, "Subsample 1 in 10\n(Effective: 3 FPS)", subtext="Idle power saver")
    draw_arrow(49.5 + 16.0/2, y_s1, 68.5 - 18.0/2, y_s1)

    # 4. Distilled NMS-Free YOLO26s
    draw_box(68.5, y_s1, 18.0, h_s1, "Distilled YOLO26s\n(NMS-Free Hungarian)", subtext="11.18 ms edge (384x640)")
    draw_arrow(68.5 + 18.0/2, y_s1, 88.5 - 17.0/2, y_s1)

    # 5. Weapon Detected?
    w_d1, h_d1 = 17.0, 12.0
    draw_diamond(88.5, y_s1, w_d1, h_d1, "Weapon\nDetected?\n(Conf > 0.45)", font_sz=9.5)

    # Diamond 1 -> NO (Loopback)
    draw_polyline_arrow([(88.5, y_s1 + h_d1/2), (88.5, 92.5), (49.5, 92.5), (49.5, y_s1 + h_s1/2)])
    ax.text(90.2, 91.0, "NO (Idle)", ha='left', va='center', fontsize=9.5, fontweight='bold', color='#14532d', zorder=4)

    # Diamond 1 -> YES
    draw_arrow(88.5 + w_d1/2, y_s1, 107.0 - 18.0/2, y_s1)
    ax.text(98.5, y_s1 + 1.2, "YES", ha='center', va='bottom', fontsize=9.8, fontweight='bold', color='#0f172a', zorder=4)

    # 6. Stage 1-Guard: Contact HOI (Optional High-Sensitivity)
    draw_box(107.0, y_s1, 18.0, h_s1, "Stage 1-Guard: HOI\n(DINOv2 Grasp)", subtext="Arm proximity + Grasp affinity")
    draw_arrow(107.0 + 18.0/2, y_s1, 128.0 - 18.0/2, y_s1)

    # 7. Trigger Stage 2
    draw_box(128.0, y_s1, 18.0, h_s1, "Lock Frame Index\n& Trigger Stage 2", subtext="Audit worker awakened", box_type='champ')

    # Connecting Arrow from Stage 1 to Stage 2a
    draw_polyline_arrow([(128.0, y_s1 - h_s1/2), (128.0, 66.5), (14.0, 66.5), (14.0, 56.5 + 8.2/2)], color='#1d4ed8', lw=2.0)

    # =========================================================================
    # STAGE 2A: MULTI-PERSON TRACKING & KINEMATICS
    # =========================================================================
    s2a_rect = Rectangle((2.0, 46.5), 144.0, 20.5,
                         facecolor=C_STAGE2A_BG, edgecolor=C_STAGE2A_BORDER,
                         linestyle='--', linewidth=1.8, zorder=1)
    ax.add_patch(s2a_rect)
    ax.text(74.0, 64.5, "STAGE 2a: Multi-Person Tracking & Biomechanical Invariance (Gapless Auditing)",
            ha='center', va='center', fontsize=12.5, fontweight='bold', family='sans-serif', color='#1e3a8a', zorder=3)
    ax.text(4.0, 48.2, "ByteTrack threat lock eliminates ID swaps (0 swaps) • Pose caching reuses 40/50 frames • Torso-scale normalization removes distance drift",
            ha='left', va='center', fontsize=9.2, family='sans-serif', color='#1d4ed8', zorder=3)

    y_s2a = 55.5
    h_s2a = 8.5

    # 1. 50-Frame Sliding Window
    draw_box(14.0, y_s2a, 20.0, h_s2a, "50-Frame Sliding Window\n(10-Frame Stride)", subtext="1.67s temporal window")
    draw_arrow(14.0 + 20.0/2, y_s2a, 41.5 - 23.0/2, y_s2a)

    # 2. YOLO26s-Pose + ByteTrack
    draw_box(41.5, y_s2a, 23.0, h_s2a, "YOLO26s-Pose + ByteTrack\n(Threat Actor Locking)", subtext="Pose Caching (40/50 frames reused)")
    draw_arrow(41.5 + 23.0/2, y_s2a, 71.5 - 25.0/2, y_s2a)

    # 3. Torso-Scale Normalization
    draw_box(71.5, y_s2a, 25.0, h_s2a, "Torso-Scale Normalization\n& Gaussian Filter (σ=1.0)", subtext="L_torso = ||shoulder - hip||_2")
    draw_arrow(71.5 + 25.0/2, y_s2a, 104.0 - 26.0/2, y_s2a)

    # 4. Sequence Max-Pooling Dispatcher
    draw_box(104.0, y_s2a, 26.0, h_s2a, "Sequence Temporal Max-Pool\nDispatcher: P_tier,max", subtext="Prevents pre-attack walking drops", box_type='champ')

    # Connecting Arrow from Stage 2a to Stage 2b
    draw_polyline_arrow([(104.0, y_s2a - h_s2a/2), (104.0, 44.0), (16.0, 44.0), (16.0, 34.0)], color='#7e22ce', lw=2.0)

    # =========================================================================
    # STAGE 2B: PROGRESSIVE MULTI-TIER STAIRCASE CASCADE
    # =========================================================================
    s2b_rect = Rectangle((2.0, 2.0), 144.0, 42.5,
                         facecolor=C_STAGE2B_BG, edgecolor=C_STAGE2B_BORDER,
                         linestyle='--', linewidth=1.8, zorder=1)
    ax.add_patch(s2b_rect)
    ax.text(74.0, 42.0, "STAGE 2b: Progressive Multi-Tier 'Staircase' Cascade (Production Champion 2: pc3d_t3 → pc3d_dt3 → stgcn)",
            ha='center', va='center', fontsize=13.0, fontweight='bold', family='sans-serif', color='#581c87', zorder=3)
    ax.text(4.0, 3.8, "Early-Exit Topology: 54.55% exits at Tier 1 • 23.38% exits at Tier 2 (77.92% offloaded from ST-GCN) • Zero-copy UMA memory reuse cuts latency by 30.3%",
            ha='left', va='center', fontsize=9.2, family='sans-serif', color='#6b21a8', zorder=3)

    # TIER 1 (3D CNN)
    x_t1 = 18.0
    y_t1_box = 28.5
    draw_box(x_t1, y_t1_box, 24.0, 8.5, "TIER 1: PoseConv3D T3\n(598k params, 3D CNN)", subtext="3D Heatmap (17, 50, 56, 56)")
    draw_arrow(x_t1, y_t1_box - 8.5/2, x_t1, 16.5 + 5.5)

    draw_diamond(x_t1, 16.5, 17.0, 11.0, "P1,max < 0.20?\n(Civilian)", font_sz=9.2)
    # Tier 1 Discard
    draw_arrow(x_t1, 16.5 - 5.5, x_t1, 7.5 + 4.5/2)
    ax.text(x_t1 + 1.2, 11.8, "YES", ha='left', va='center', fontsize=9.0, fontweight='bold', color='#14532d', zorder=4)
    draw_box(x_t1, 7.5, 20.0, 4.5, "FAST DISCARD\n(54.55% Resolved)", box_type='dismiss', font_sz=8.8)

    # Tier 1 Escalate to Tier 2
    draw_polyline_arrow([(x_t1 + 17.0/2, 16.5), (37.0, 16.5), (37.0, 28.5), (48.0 - 24.0/2, 28.5)], color='#7e22ce', lw=1.8)
    ax.text(37.5, 23.0, "ESCALATE (45.5%)\n[Zero-Copy Heatmap]", ha='left', va='center', fontsize=8.6, fontweight='bold', color='#6b21a8', zorder=4)

    # TIER 2 (Distilled 3D CNN)
    x_t2 = 58.0
    draw_box(x_t2, y_t1_box, 25.0, 8.5, "TIER 2: PoseConv3D DT3\n(598k params, RKD Student)", subtext="Zero-Copy GPU UMA Heatmap", box_type='champ')
    draw_arrow(x_t2, y_t1_box - 8.5/2, x_t2, 16.5 + 5.5)

    draw_diamond(x_t2, 16.5, 17.0, 11.0, "P2,max < 0.35?\n(Civilian)", font_sz=9.2)
    # Tier 2 Discard
    draw_arrow(x_t2, 16.5 - 5.5, x_t2, 7.5 + 4.5/2)
    ax.text(x_t2 + 1.2, 11.8, "YES", ha='left', va='center', fontsize=9.0, fontweight='bold', color='#14532d', zorder=4)
    draw_box(x_t2, 7.5, 20.0, 4.5, "FAST DISCARD\n(23.38% Resolved)", box_type='dismiss', font_sz=8.8)

    # Tier 2 Escalate to Tier 3
    draw_polyline_arrow([(x_t2 + 17.0/2, 16.5), (78.5, 16.5), (78.5, 28.5), (90.0 - 24.0/2, 28.5)], color='#7e22ce', lw=1.8)
    ax.text(79.0, 23.0, "ESCALATE (22.1%)\n[Load 2D Coords]", ha='left', va='center', fontsize=8.6, fontweight='bold', color='#6b21a8', zorder=4)

    # TIER 3 (9-Block ST-GCN)
    x_t3 = 99.0
    draw_box(x_t3, y_t1_box, 24.0, 8.5, "TIER 3: 9-Block ST-GCN\n(3.01M params, 2D Graph)", subtext="Only evaluates 22.08% traffic")
    draw_arrow(x_t3, y_t1_box - 8.5/2, x_t3, 16.5 + 5.5)

    draw_diamond(x_t3, 16.5, 17.0, 11.0, "P3,max >= 0.55?\n(Assault)", font_sz=9.2)
    # Tier 3 Discard
    draw_arrow(x_t3, 16.5 - 5.5, x_t3, 7.5 + 4.5/2)
    ax.text(x_t3 + 1.2, 11.8, "NO", ha='left', va='center', fontsize=9.0, fontweight='bold', color='#14532d', zorder=4)
    draw_box(x_t3, 7.5, 20.0, 4.5, "DISMISS AS CIVILIAN\n(Final Safe Exit)", box_type='dismiss', font_sz=8.8)

    # Tier 3 Alarm -> Violence Alarm Box
    x_alarm = 132.5
    draw_arrow(x_t3 + 17.0/2, 16.5, x_alarm - 22.0/2, 16.5, color='#b91c1c', lw=2.2)
    ax.text(114.5, 18.2, "YES (Assault)", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#991b1b', zorder=4)

    draw_box(x_alarm, 16.5, 23.0, 12.5, "HIGH-CONFIDENCE ALARM\n[Active Assault Verified]\n1.0000 F1 | 100% Recall\n0 FP (0 / 44 civilian)", box_type='violence', font_sz=10.2)
    ax.text(x_alarm, 7.2, "34.80 FPS sustained GPU\n5.24 ms action latency", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#991b1b', zorder=3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, facecolor='#ffffff', edgecolor='none')
    plt.close()
    print('Render completed: ' + output_path)

if __name__ == '__main__':
    create_pipeline_diagram('assets/inference_pipeline.png', dpi=200)
