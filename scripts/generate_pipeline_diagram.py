import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import os

def create_pipeline_diagram(output_path='assets/inference_pipeline.png', dpi=200):
    output_path = os.path.normpath(os.path.abspath(output_path))
    fig = plt.figure(figsize=(17.8, 9.4), dpi=dpi, facecolor='#ffffff')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 142.4)
    ax.set_ylim(0, 75.2)
    ax.axis('off')
    
    # Palette definition
    C_OUTSIDE_BG = '#f7fafb'       # Ultra-clean canvas background
    C_STAGE_BG = '#edf5f6'         # Subtle tinted container background
    C_STAGE_BORDER = '#254e5c'     # Dashed stage boundary
    C_BOX_BG = '#ffffff'           # Process box fill
    C_BOX_BORDER = '#1b4350'       # Process box border
    C_DIAMOND_BG = '#ffffff'       # Diamond fill
    C_DIAMOND_BORDER = '#1b4350'   # Diamond border
    C_TEXT = '#0f242c'             # Primary text
    C_NOTE_TEXT = '#1e3c47'        # Subtitle / note text
    C_ARROW = '#1b4350'            # Arrow and routing lines
    
    # Violence classification box styling
    C_VIOLENCE_BG = '#fee2e2'      # Soft red fill for violence classification
    C_VIOLENCE_BORDER = '#b91c1c'  # Crimson border
    C_VIOLENCE_TEXT = '#7f1d1d'    # Crimson text
    
    # Non-threat dismissal styling
    C_DISMISS_BG = '#f0fdf4'       # Soft mint/green fill for non-threat
    C_DISMISS_BORDER = '#15803d'   # Muted green border
    C_DISMISS_TEXT = '#14532d'     # Dark green text

    # Base canvas fill
    ax.add_patch(Rectangle((0, 0), 142.4, 75.2, facecolor=C_OUTSIDE_BG, edgecolor='none', zorder=0))

    def draw_box(x, y, w, h, title_text, box_type='standard'):
        if box_type == 'violence':
            fc, ec, tc, lw = C_VIOLENCE_BG, C_VIOLENCE_BORDER, C_VIOLENCE_TEXT, 2.2
        elif box_type == 'dismiss':
            fc, ec, tc, lw = C_DISMISS_BG, C_DISMISS_BORDER, C_DISMISS_TEXT, 1.8
        else:
            fc, ec, tc, lw = C_BOX_BG, C_BOX_BORDER, C_TEXT, 1.8
            
        rect = Rectangle((x - w/2, y - h/2), w, h, 
                         facecolor=fc, edgecolor=ec, linewidth=lw, 
                         zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, title_text, ha='center', va='center', 
                fontsize=11.2, fontweight='bold', family='sans-serif',
                color=tc, linespacing=1.28, zorder=3)
        return (x, y, w, h)

    def draw_diamond(x, y, w, h, text, font_sz=10.5):
        vertices = [(x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)]
        poly = Polygon(vertices, closed=True, facecolor=C_DIAMOND_BG, 
                       edgecolor=C_DIAMOND_BORDER, linewidth=1.8, zorder=2)
        ax.add_patch(poly)
        ax.text(x, y, text, ha='center', va='center',
                fontsize=font_sz, fontweight='bold', family='sans-serif',
                color=C_TEXT, linespacing=1.25, zorder=3)

    def draw_arrow(x1, y1, x2, y2, zorder=4):
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='-|>',
                                mutation_scale=15,
                                linewidth=1.8,
                                color=C_ARROW,
                                zorder=zorder)
        ax.add_patch(arrow)

    def draw_polyline_arrow(points, zorder=4):
        for i in range(len(points) - 2):
            ax.add_line(Line2D([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]],
                               color=C_ARROW, linewidth=1.8, zorder=zorder))
        p_prev = points[-2]
        p_last = points[-1]
        draw_arrow(p_prev[0], p_prev[1], p_last[0], p_last[1], zorder=zorder)

    # =========================================================================
    # STAGE 1 CONTAINER
    # =========================================================================
    s1_rect = Rectangle((1.5, 39.5), 139.4, 33.5,
                        facecolor=C_STAGE_BG, edgecolor=C_STAGE_BORDER,
                        linestyle='--', linewidth=1.8, zorder=1)
    ax.add_patch(s1_rect)
    ax.text(71.2, 70.4, "Stage 1: Continuous Weapon Scanning (Always-On Loop)",
            ha='center', va='center', fontsize=14.5, fontweight='bold',
            family='sans-serif', color='#0a222a', zorder=3)
    ax.text(3.5, 41.5, "(Note: This entire stage runs continuously and in parallel with Stage 2)",
            ha='left', va='center', fontsize=10.5, family='sans-serif',
            color=C_NOTE_TEXT, zorder=3)

    y_s1 = 59.2
    h_s1 = 9.0
    
    # 1. Video Ingestion
    draw_box(11.5, y_s1, 16.5, h_s1, "[Live Video Stream\n(30 FPS)]")
    draw_arrow(11.5 + 16.5/2, y_s1, 30.5 - 15.5/2, y_s1)

    # 2. Circular Buffer
    draw_box(30.5, y_s1, 15.5, h_s1, "[Store Frame in\nCircular Buffer]")
    draw_arrow(30.5 + 15.5/2, y_s1, 50.0 - 15.5/2, y_s1)

    # 3. 3 FPS Subsampling
    draw_box(50.0, y_s1, 15.5, h_s1, "[Sample 1 Frame\nper 10 (3 FPS)]")
    draw_arrow(50.0 + 15.5/2, y_s1, 70.0 - 16.0/2, y_s1)

    # 4. Distilled YOLO26s Weapon Detector
    draw_box(70.0, y_s1, 16.0, h_s1, "[Distilled YOLO26s\nWeapon Detection]")
    draw_arrow(70.0 + 16.0/2, y_s1, 91.5 - 16.5/2, y_s1)

    # 5. Diamond 1: Conf > 0.45
    w_d1, h_d1 = 16.5, 12.8
    draw_diamond(91.5, y_s1, w_d1, h_d1, "<Weapon\nDetected?\n(Conf > 0.45)>", font_sz=10.5)

    # Diamond 1 -> NO Loopback
    draw_polyline_arrow([(91.5, y_s1 + h_d1/2), (91.5, 67.8), (50.0, 67.8), (50.0, y_s1 + h_s1/2)])
    ax.text(93.0, 66.2, "NO", ha='left', va='center', fontsize=11, fontweight='bold',
            family='sans-serif', color='#0f242c', zorder=4)

    # Diamond 1 -> YES (Down to Box 1_5)
    draw_arrow(91.5, y_s1 - h_d1/2, 91.5, 46.0 + 8.2/2)
    ax.text(93.0, 50.8, "YES", ha='left', va='center', fontsize=11, fontweight='bold',
            family='sans-serif', color='#0f242c', zorder=4)

    # 6. Mark frame & trigger Stage 2
    draw_box(91.5, 46.0, 18.0, 8.2, "[Mark the frame index\nand Trigger Stage 2]")

    # =========================================================================
    # STAGE 1 TO STAGE 2 CONNECTING ARROW
    # =========================================================================
    draw_polyline_arrow([(91.5, 46.0 - 8.2/2), (91.5, 36.8), (11.5, 36.8), (11.5, 24.2 + 9.5/2)])

    # =========================================================================
    # STAGE 2 CONTAINER
    # =========================================================================
    s2_rect = Rectangle((1.5, 1.8), 139.4, 33.2,
                        facecolor=C_STAGE_BG, edgecolor=C_STAGE_BORDER,
                        linestyle='--', linewidth=1.8, zorder=1)
    ax.add_patch(s2_rect)
    ax.text(71.2, 32.5, "Stage 2: On-Demand Action Analysis (Triggered Process)",
            ha='center', va='center', fontsize=14.5, fontweight='bold',
            family='sans-serif', color='#0a222a', zorder=3)

    y_s2 = 24.2
    h_s2 = 9.5

    # 1. 50-Frame Sliding Window
    draw_box(11.5, y_s2, 16.5, h_s2, "[Create/Extend\n50-Frame Analysis\nWindow]")
    ax.text(11.5, 14.5, "(Note: If a new\nweapon is detected, this\nwindow is extended)",
            ha='center', va='center', fontsize=9.8, family='sans-serif',
            color=C_NOTE_TEXT, linespacing=1.2, zorder=3)

    draw_arrow(11.5 + 16.5/2, y_s2, 29.5 - 16.0/2, y_s2)

    # 2. Skeleton Extraction via YOLO26s-Pose
    draw_box(29.5, y_s2, 16.0, h_s2, "[Extract Skeletons\nvia YOLO26s-Pose\n(Keypoint Caching)]")
    draw_arrow(29.5 + 16.0/2, y_s2, 47.5 - 15.5/2, y_s2)

    # 3. Kinematic Preprocessing & Normalization
    draw_box(47.5, y_s2, 15.5, h_s2, "[Kinematic\nSmoothing &\nNormalization]")
    draw_arrow(47.5 + 15.5/2, y_s2, 65.5 - 15.0/2, y_s2)

    # 4. ST-GCN Feature Extraction
    draw_box(65.5, y_s2, 15.0, h_s2, "[ST-GCN\nFeature\nExtraction\n(128-d Vector)]")
    draw_arrow(65.5 + 15.0/2, y_s2, 84.5 - 17.0/2, y_s2)

    # 5. Deep k-NN Distance
    draw_box(84.5, y_s2, 17.0, h_s2, "[Calculate\nDeep k-NN\nDistance (k=2 vs\nThreat Bank)]")
    draw_arrow(84.5 + 17.0/2, y_s2, 105.0 - 16.5/2, y_s2)

    # 6. Diamond 2: Distance <= tau (tau = 3.48)
    w_d2, h_d2 = 16.5, 12.8
    draw_diamond(105.0, y_s2, w_d2, h_d2, "<Any Clip's\nDistance <= tau?\n(tau = 3.48)>", font_sz=10.2)

    # Diamond 2 -> NO (Dismiss as Non-Threat)
    draw_arrow(105.0 + w_d2/2, y_s2, 128.0 - 17.5/2, y_s2)
    ax.text(115.5, y_s2 + 1.2, "NO", ha='center', va='bottom', fontsize=11, fontweight='bold',
            family='sans-serif', color='#0f242c', zorder=4)

    draw_box(128.0, y_s2, 17.5, h_s2, "[Dismiss Event as\nNon-Threat]", box_type='dismiss')
    ax.text(128.0, 16.5, "(Passive Threat: Safe;\nProcess ends)",
            ha='center', va='center', fontsize=9.8, family='sans-serif',
            color=C_NOTE_TEXT, linespacing=1.2, zorder=3)

    # Diamond 2 -> YES (Violence Classification - Pipeline Ends Here)
    y_violence = 8.5
    h_violence = 8.5
    draw_arrow(105.0, y_s2 - h_d2/2, 105.0, y_violence + h_violence/2)
    ax.text(105.0, 15.2, "YES", ha='center', va='center', fontsize=11, fontweight='bold',
            family='sans-serif', color='#0f242c', zorder=4)

    draw_box(105.0, y_violence, 21.0, h_violence, "[Classify Event as\n\"Violence\"]", box_type='violence')
    ax.text(105.0, 2.7, "(Active Threat Verified; Process ends)",
            ha='center', va='center', fontsize=9.5, family='sans-serif',
            color=C_NOTE_TEXT, zorder=3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, facecolor='#ffffff', edgecolor='none')
    plt.close()
    print('Render completed: ' + output_path)

if __name__ == '__main__':
    create_pipeline_diagram('assets/inference_pipeline.png', dpi=200)
