import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import os

def create_pipeline_diagram(output_path='assets/inference_pipeline.png', dpi=200):
    output_path = os.path.normpath(os.path.abspath(output_path))
    fig = plt.figure(figsize=(20.0, 13.2), dpi=dpi, facecolor='#ffffff')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 150.0)
    ax.set_ylim(0, 106.0)
    ax.axis('off')
    
    # Palette definition
    C_OUTSIDE_BG = '#f8fafc'       # Modern slate canvas
    C_STAGE1_BG = '#f0fdf4'        # Soft emerald tint (Stage 1)
    C_STAGE1_BORDER = '#16a34a'    # Emerald border
    C_STAGE2A_BG = '#eff6ff'       # Soft sky blue tint (Stage 2a)
    C_STAGE2A_BORDER = '#2563eb'   # Blue border
    C_STAGE2B_BG = '#faf5ff'       # Soft purple tint (Stage 2b)
    C_STAGE2B_BORDER = '#9333ea'   # Purple border
    
    C_BOX_BG = '#ffffff'           # Process box fill
    C_BOX_BORDER = '#334155'       # Dark slate border
    C_DIAMOND_BG = '#ffffff'       # Decision diamond fill
    C_DIAMOND_BORDER = '#1e293b'   # Decision diamond border
    C_TEXT = '#0f172a'             # Primary text
    C_NOTE_TEXT = '#475569'        # Note/annotation text
    C_ARROW = '#1e293b'            # Arrow lines
    
    # Alert & Badge styling
    C_VIOLENCE_BG = '#fee2e2'      # Soft red fill
    C_VIOLENCE_BORDER = '#b91c1c'  # Crimson border
    C_VIOLENCE_TEXT = '#991b1b'    # Crimson text
    
    C_DISMISS_BG = '#f1f5f9'       # Neutral slate fill
    C_DISMISS_BORDER = '#64748b'   # Muted border
    C_DISMISS_TEXT = '#334155'     # Muted text
    
    C_CHAMP_BG = '#fef3c7'         # Amber champion badge fill
    C_CHAMP_BORDER = '#d97706'     # Amber border
    C_CHAMP_TEXT = '#92400e'       # Amber text

    # Base canvas background
    ax.add_patch(Rectangle((0, 0), 150.0, 106.0, facecolor=C_OUTSIDE_BG, edgecolor='none', zorder=0))

    def draw_box(x, y, w, h, title_text, box_type='standard', font_sz=10.2, subtext=None, title_color=None, sub_font_sz=8.4):
        if box_type == 'violence':
            fc, ec, tc, lw = C_VIOLENCE_BG, C_VIOLENCE_BORDER, C_VIOLENCE_TEXT, 2.4
        elif box_type == 'dismiss':
            fc, ec, tc, lw = C_DISMISS_BG, C_DISMISS_BORDER, C_DISMISS_TEXT, 1.6
        elif box_type == 'champ':
            fc, ec, tc, lw = C_CHAMP_BG, C_CHAMP_BORDER, C_CHAMP_TEXT, 2.0
        else:
            fc, ec, tc, lw = C_BOX_BG, C_BOX_BORDER, C_TEXT, 1.8
            
        if title_color:
            tc = title_color
            
        rect = Rectangle((x - w/2, y - h/2), w, h, 
                         facecolor=fc, edgecolor=ec, linewidth=lw, 
                         zorder=2)
        ax.add_patch(rect)
        
        if subtext:
            ax.text(x, y + 1.2, title_text, ha='center', va='center', 
                    fontsize=font_sz, fontweight='bold', family='sans-serif',
                    color=tc, linespacing=1.2, zorder=3)
            ax.text(x, y - h/2 + 1.4, subtext, ha='center', va='center',
                    fontsize=sub_font_sz, family='sans-serif', color=C_NOTE_TEXT, zorder=3)
        else:
            ax.text(x, y, title_text, ha='center', va='center', 
                    fontsize=font_sz, fontweight='bold', family='sans-serif',
                    color=tc, linespacing=1.22, zorder=3)
        return (x, y, w, h)

    def draw_diamond(x, y, w, h, text, font_sz=9.5):
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
    ax.text(75.0, 102.5, "Hierarchical Edge-AI Surveillance: End-to-End Threat & Violence Detection Cascade",
            ha='center', va='center', fontsize=15.5, fontweight='heavy', family='sans-serif', color='#0f172a')

    # =========================================================================
    # STAGE 1: ALWAYS-ON WEAPON SCREENING
    # =========================================================================
    s1_rect = Rectangle((2.5, 71.0), 145.0, 28.0,
                        facecolor=C_STAGE1_BG, edgecolor=C_STAGE1_BORDER,
                        linestyle='--', linewidth=1.8, zorder=1)
    ax.add_patch(s1_rect)
    ax.text(75.0, 96.8, "STAGE 1: Continuous Lightweight Threat Screening (Always-On Ingestion & Gating Loop)",
            ha='center', va='center', fontsize=12.5, fontweight='bold', family='sans-serif', color='#14532d', zorder=3)
    ax.text(4.5, 72.8, "Subsamples 1 in 10 frames (3 FPS) • Cuts idle GPU workloads by ~90%",
            ha='left', va='center', fontsize=9.0, family='sans-serif', color='#166534', zorder=3)

    y_s1 = 84.5
    h_s1 = 8.5

    # 1. Live Video Stream (X=14.0, W=16)
    draw_box(14.0, y_s1, 16.0, h_s1, "Live Surveillance\nStream (30 FPS)", subtext="RTSP / USB Camera")
    draw_arrow(14.0 + 16.0/2, y_s1, 36.0 - 19.0/2, y_s1)

    # 2. TurboJPEG Circular Buffer (X=36.0, W=19)
    draw_box(36.0, y_s1, 19.0, h_s1, "TurboJPEG SIMD\nCircular Buffer", subtext="1200 buffer frames as ~188.5 MB", sub_font_sz=7.2)
    draw_arrow(36.0 + 19.0/2, y_s1, 58.0 - 16.0/2, y_s1)

    # 3. 3 FPS Subsampling (X=58.0, W=16)
    draw_box(58.0, y_s1, 16.0, h_s1, "Subsample 1 in 10\n(Effective: 3 FPS)", subtext="Idle power saver")
    draw_arrow(58.0 + 16.0/2, y_s1, 80.0 - 18.0/2, y_s1)

    # 4. Distilled NMS-Free YOLO26s (X=80.0, W=18)
    draw_box(80.0, y_s1, 18.0, h_s1, "Distilled YOLO26s\n(NMS-Free Hungarian)", subtext=None)
    draw_arrow(80.0 + 18.0/2, y_s1, 104.0 - 17.0/2, y_s1)

    # 5. Weapon Detected? (X=104.0, W=17, H=11.5)
    w_d1, h_d1 = 17.0, 11.5
    draw_diamond(104.0, y_s1, w_d1, h_d1, "Weapon\nDetected?\n(Conf > 0.45)", font_sz=9.2)

    # Diamond 1 -> NO (Loopback to 3 FPS Subsampling)
    draw_polyline_arrow([(104.0, y_s1 + h_d1/2), (104.0, 94.0), (58.0, 94.0), (58.0, y_s1 + h_s1/2)])
    ax.text(105.5, 92.5, "NO (Idle)", ha='left', va='center', fontsize=9.2, fontweight='bold', color='#14532d', zorder=4)

    # Diamond 1 -> YES (Direct Trigger to Stage 2)
    draw_arrow(104.0 + w_d1/2, y_s1, 131.0 - 22.0/2, y_s1)
    ax.text(116.25, y_s1 + 1.2, "YES", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#0f172a', zorder=4)

    # 6. Trigger Stage 2 (X=131.0, W=22)
    draw_box(131.0, y_s1, 22.0, h_s1, "Lock Frame Index\n& Trigger Stage 2", subtext="Awakens Stage 2 Analysis", box_type='champ')

    # Connecting Arrow from Stage 1 to Stage 2a
    draw_polyline_arrow([(131.0, y_s1 - h_s1/2), (131.0, 69.8), (17.5, 69.8), (17.5, 57.5 + 8.5/2)], color='#2563eb', lw=2.0)

    # =========================================================================
    # STAGE 2A: MULTI-PERSON TRACKING & KINEMATICS
    # =========================================================================
    s2a_rect = Rectangle((2.5, 48.0), 145.0, 20.5,
                         facecolor=C_STAGE2A_BG, edgecolor=C_STAGE2A_BORDER,
                         linestyle='--', linewidth=1.8, zorder=1)
    ax.add_patch(s2a_rect)
    ax.text(75.0, 66.2, "STAGE 2a: Multi-Person Tracking & Biomechanical Invariance (Gapless Auditing)",
            ha='center', va='center', fontsize=12.5, fontweight='bold', family='sans-serif', color='#1e3a8a', zorder=3)
    ax.text(4.5, 49.6, "ByteTrack persistent threat lock • Pose caching reuses 40/50 frames • Torso-scale normalization removes distance drift",
            ha='left', va='center', fontsize=9.0, family='sans-serif', color='#1d4ed8', zorder=3)

    y_s2a = 57.5
    h_s2a = 8.5

    # 1. 50-Frame Sliding Window (X=17.5, W=25)
    draw_box(17.5, y_s2a, 25.0, h_s2a, "50-Frame Sliding Window\n(10-Frame Stride)", subtext=None)
    draw_arrow(17.5 + 25.0/2, y_s2a, 55.8 - 25.0/2, y_s2a)

    # 2. YOLO26s-Pose + ByteTrack (X=55.8, W=25)
    draw_box(55.8, y_s2a, 25.0, h_s2a, "YOLO26s-Pose + ByteTrack\n(Threat Actor Locking)", subtext="Pose Caching (40/50 frames reused)")
    draw_arrow(55.8 + 25.0/2, y_s2a, 94.1 - 25.0/2, y_s2a)

    # 3. Torso-Scale Normalization (X=94.1, W=25)
    draw_box(94.1, y_s2a, 25.0, h_s2a, "Torso-Scale Normalization\n& Gaussian Filter (σ=1.0)", subtext="L_torso = ||shoulder - hip||_2")
    draw_arrow(94.1 + 25.0/2, y_s2a, 133.4 - 27.0/2, y_s2a)

    # 4. Dual-Representation Formatter (X=133.4, W=27)
    draw_box(133.4, y_s2a, 27.0, h_s2a, "Dual-Representation Formatter\n(2D Graph & 3D Heatmaps)", subtext=None, box_type='champ')

    # Connecting Arrow from Stage 2a to Stage 2b
    draw_polyline_arrow([(133.4, y_s2a - h_s2a/2), (133.4, 45.5), (17.5, 45.5), (17.5, 33.0 + 8.5/2)], color='#9333ea', lw=2.0)

    # =========================================================================
    # STAGE 2B: PROGRESSIVE MULTI-TIER STAIRCASE CASCADE
    # =========================================================================
    s2b_rect = Rectangle((2.5, 2.0), 145.0, 43.5,
                         facecolor=C_STAGE2B_BG, edgecolor=C_STAGE2B_BORDER,
                         linestyle='--', linewidth=1.8, zorder=1)
    ax.add_patch(s2b_rect)
    ax.text(75.0, 43.2, "STAGE 2b: Progressive Multi-Tier 'Staircase' Cascade",
            ha='center', va='center', fontsize=12.5, fontweight='bold', family='sans-serif', color='#581c87', zorder=3)
    ax.text(4.5, 3.8, "Dynamic Early-Exit Architecture: Benign actions exit early at Tier 1/2 • Heavy ST-GCN invoked only on ambiguous escalation • Zero-copy GPU UMA memory reuse",
            ha='left', va='center', fontsize=9.0, family='sans-serif', color='#6b21a8', zorder=3)

    # Four columns: X = 17.5, 55.8, 94.1, 133.4
    y_box_s2b = 33.0
    h_box_s2b = 8.5
    y_diamond_s2b = 18.5
    w_diamond_s2b = 16.0
    h_diamond_s2b = 10.5
    y_discard_s2b = 7.5
    h_discard_s2b = 5.2

    # -------------------------------------------------------------------------
    # COL 1: TIER 1 (PoseConv3D Tier 3)
    # -------------------------------------------------------------------------
    x_c1 = 17.5
    draw_box(x_c1, y_box_s2b, 25.0, h_box_s2b, "TIER 1: PoseConv3D T3\n(598k params, 3D CNN)", subtext="3D Heatmap (17, 50, 56, 56)")
    draw_arrow(x_c1, y_box_s2b - h_box_s2b/2, x_c1, y_diamond_s2b + h_diamond_s2b/2)

    draw_diamond(x_c1, y_diamond_s2b, w_diamond_s2b, h_diamond_s2b, "P1,max < 0.20?\n(Non-Threat)", font_sz=9.0)

    # Discard Down
    draw_arrow(x_c1, y_diamond_s2b - h_diamond_s2b/2, x_c1, y_discard_s2b + h_discard_s2b/2)
    ax.text(x_c1 + 1.2, 13.0, "YES", ha='left', va='center', fontsize=8.8, fontweight='bold', color='#14532d', zorder=4)
    draw_box(x_c1, y_discard_s2b, 21.0, h_discard_s2b, "FAST DISCARD\n(Early Safe Exit)", box_type='dismiss', font_sz=8.6)

    # Escalation from Tier 1 Diamond (right vertex: x=25.5, y=18.5) to Tier 2 Box (left edge: x=43.3, y=33.0)
    x_t1_d_right = x_c1 + w_diamond_s2b/2      # 25.5
    x_t2_b_left = 55.8 - 25.0/2               # 43.3
    x_mid_12 = (x_t1_d_right + x_t2_b_left) / 2 # 34.4

    draw_polyline_arrow([(x_t1_d_right, y_diamond_s2b),
                         (x_mid_12, y_diamond_s2b),
                         (x_mid_12, y_box_s2b),
                         (x_t2_b_left, y_box_s2b)], color='#7e22ce', lw=2.0)

    # Escalation Label neatly positioned in the channel
    ax.text(x_mid_12 - 0.8, (y_diamond_s2b + y_box_s2b)/2, "ESCALATE\n[Zero-Copy Heatmap]",
            ha='right', va='center', fontsize=8.4, fontweight='bold', color='#6b21a8', linespacing=1.2, zorder=4)
    ax.text(x_t1_d_right + 1.0, y_diamond_s2b + 1.2, "NO",
            ha='left', va='bottom', fontsize=8.8, fontweight='bold', color='#6b21a8', zorder=4)

    # -------------------------------------------------------------------------
    # COL 2: TIER 2 (PoseConv3D Distilled T3)
    # -------------------------------------------------------------------------
    x_c2 = 55.8
    draw_box(x_c2, y_box_s2b, 25.0, h_box_s2b, "TIER 2: PoseConv3D DT3\n(598k params, RKD Student)", subtext="Zero-Copy GPU UMA Heatmap", box_type='champ')
    draw_arrow(x_c2, y_box_s2b - h_box_s2b/2, x_c2, y_diamond_s2b + h_diamond_s2b/2)

    draw_diamond(x_c2, y_diamond_s2b, w_diamond_s2b, h_diamond_s2b, "P2,max < 0.35?\n(Non-Threat)", font_sz=9.0)

    # Discard Down
    draw_arrow(x_c2, y_diamond_s2b - h_diamond_s2b/2, x_c2, y_discard_s2b + h_discard_s2b/2)
    ax.text(x_c2 + 1.2, 13.0, "YES", ha='left', va='center', fontsize=8.8, fontweight='bold', color='#14532d', zorder=4)
    draw_box(x_c2, y_discard_s2b, 21.0, h_discard_s2b, "FAST DISCARD\n(Early Safe Exit)", box_type='dismiss', font_sz=8.6)

    # Escalation from Tier 2 Diamond (right vertex: x=63.8, y=18.5) to Tier 3 Box (left edge: x=81.6, y=33.0)
    x_t2_d_right = x_c2 + w_diamond_s2b/2      # 63.8
    x_t3_b_left = 94.1 - 25.0/2               # 81.6
    x_mid_23 = (x_t2_d_right + x_t3_b_left) / 2 # 72.7

    draw_polyline_arrow([(x_t2_d_right, y_diamond_s2b),
                         (x_mid_23, y_diamond_s2b),
                         (x_mid_23, y_box_s2b),
                         (x_t3_b_left, y_box_s2b)], color='#7e22ce', lw=2.0)

    # Escalation Label neatly positioned in the channel
    ax.text(x_mid_23 - 0.8, (y_diamond_s2b + y_box_s2b)/2, "ESCALATE\n[Load 2D Graph Coords]",
            ha='right', va='center', fontsize=8.4, fontweight='bold', color='#6b21a8', linespacing=1.2, zorder=4)
    ax.text(x_t2_d_right + 1.0, y_diamond_s2b + 1.2, "NO",
            ha='left', va='bottom', fontsize=8.8, fontweight='bold', color='#6b21a8', zorder=4)

    # -------------------------------------------------------------------------
    # COL 3: TIER 3 (9-Block ST-GCN)
    # -------------------------------------------------------------------------
    x_c3 = 94.1
    draw_box(x_c3, y_box_s2b, 25.0, h_box_s2b, "TIER 3: 9-Block ST-GCN\n(3.01M params, 2D Graph)", subtext="Deep Spatio-Temporal Graph Analysis")
    draw_arrow(x_c3, y_box_s2b - h_box_s2b/2, x_c3, y_diamond_s2b + h_diamond_s2b/2)

    draw_diamond(x_c3, y_diamond_s2b, w_diamond_s2b, h_diamond_s2b, "P3,max >= 0.55?\n(Assault Confirmed)", font_sz=9.0)

    # Dismiss Down
    draw_arrow(x_c3, y_diamond_s2b - h_diamond_s2b/2, x_c3, y_discard_s2b + h_discard_s2b/2)
    ax.text(x_c3 + 1.2, 13.0, "NO", ha='left', va='center', fontsize=8.8, fontweight='bold', color='#14532d', zorder=4)
    draw_box(x_c3, y_discard_s2b, 21.0, h_discard_s2b, "DISMISS AS BENIGN\n(Final Safe Exit)", box_type='dismiss', font_sz=8.6)

    # -------------------------------------------------------------------------
    # COL 4: HIGH-CONFIDENCE VIOLENCE ALARM
    # -------------------------------------------------------------------------
    x_c4 = 133.4
    w_alarm = 27.0
    h_alarm = 14.0
    x_alarm_left = x_c4 - w_alarm/2 # 119.9
    x_t3_d_right = x_c3 + w_diamond_s2b/2 # 102.1

    # Straight horizontal arrow from Tier 3 Diamond to Alarm Box
    draw_arrow(x_t3_d_right, y_diamond_s2b, x_alarm_left, y_diamond_s2b, color='#b91c1c', lw=2.2)
    ax.text((x_t3_d_right + x_alarm_left)/2, y_diamond_s2b + 1.2, "YES (Assault)",
            ha='center', va='bottom', fontsize=9.2, fontweight='bold', color='#991b1b', zorder=4)

    draw_box(x_c4, y_diamond_s2b, w_alarm, 16.5,
             "HIGH-CONFIDENCE ALARM\n[Active Assault Verified]\n\n• Instant Alert Dispatch\n• Security Team Notification\n• Incident Video Archiving",
             box_type='violence', font_sz=9.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, facecolor='#ffffff', edgecolor='none')
    plt.close()
    print('Render completed: ' + output_path)

if __name__ == '__main__':
    create_pipeline_diagram('assets/inference_pipeline.png', dpi=200)
