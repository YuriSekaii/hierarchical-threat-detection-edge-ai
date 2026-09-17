"""
Generates a publication-quality, single-panel PoseConv3D Pareto Frontier chart:
Capacity Scaling vs. F1-Score with CPU Throughput (FPS) Annotated.

Replaces the redundant two-panel subplot with a clean, high-impact chart.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def generate_pareto_chart(output_path=None):
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        output_path = os.path.join(repo_root, "assets", "poseconv3d_pareto_frontier.png")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Model data
    models = [
        {
            "name": "Tier 5 (132k)",
            "short_name": "Tier 5 (132k)",
            "params": 132349,
            "f1": 0.8621,
            "fps": 29.7,
            "color": "#2ecc71",
            "marker": "o",
            "size": 110,
            "text_offset": (15, -4),
            "ha": "left",
            "va": "center",
            "badge": "29.7 FPS",
            "highlight": False
        },
        {
            "name": "Tier 4 (232k)",
            "short_name": "Tier 4 (232k)",
            "params": 232337,
            "f1": 0.8125,
            "fps": 32.5,
            "color": "#95a5a6",
            "marker": "o",
            "size": 90,
            "text_offset": (15, -2),
            "ha": "left",
            "va": "center",
            "badge": "32.5 FPS",
            "highlight": False
        },
        {
            "name": "Tier 3 (598k) [Production Champion]",
            "short_name": "Tier 3 (598k)",
            "params": 598429,
            "f1": 0.8857,
            "fps": 27.9,
            "color": "#e74c3c",
            "marker": "*",
            "size": 280,
            "text_offset": (18, 2),
            "ha": "left",
            "va": "center",
            "badge": "27.9 FPS",
            "highlight": True
        },
        {
            "name": "Tier 2 (647k)",
            "short_name": "Tier 2 (647k)",
            "params": 647185,
            "f1": 0.8108,
            "fps": 32.9,
            "color": "#95a5a6",
            "marker": "o",
            "size": 90,
            "text_offset": (-15, -16),
            "ha": "right",
            "va": "top",
            "badge": "32.9 FPS",
            "highlight": False
        },
        {
            "name": "Tier 1 (703k)",
            "short_name": "Tier 1 (703k)",
            "params": 702885,
            "f1": 0.8116,
            "fps": 28.4,
            "color": "#7f8c8d",
            "marker": "o",
            "size": 90,
            "text_offset": (15, -12),
            "ha": "left",
            "va": "top",
            "badge": "28.4 FPS",
            "highlight": False
        },
        {
            "name": "Baseline v2.10 (765k)",
            "short_name": "Baseline v2.10 (765k)",
            "params": 765529,
            "f1": 0.8667,
            "fps": 23.3,
            "color": "#2980b9",
            "marker": "s",
            "size": 100,
            "text_offset": (15, 0),
            "ha": "left",
            "va": "center",
            "badge": "23.3 FPS",
            "highlight": False
        },
        {
            "name": "Scaled (3.55M)",
            "short_name": "Scaled (3.55M)",
            "params": 3545409,
            "f1": 0.8406,
            "fps": 5.6,
            "color": "#e67e22",
            "marker": "^",
            "size": 110,
            "text_offset": (15, 0),
            "ha": "left",
            "va": "center",
            "badge": "5.6 FPS",
            "highlight": False
        }
    ]
    
    fig, ax = plt.subplots(figsize=(9.5, 6.2), dpi=300)
    
    # Set background styling
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#ffffff')
    
    # Grid
    ax.grid(True, which='both', linestyle=':', color='#e0e0e0', linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    
    # Plot baseline reference line
    baseline_f1 = 0.8667
    ax.axhline(baseline_f1, color='#3498db', linestyle='--', linewidth=1.5, alpha=0.75,
               label=f'Baseline v2.10 F1 ({baseline_f1:.4f})')
    
    # Scatter points & Annotations
    for m in models:
        # Plot marker
        edge_color = '#c0392b' if m['highlight'] else 'none'
        edge_w = 1.2 if m['highlight'] else 0
        ax.scatter(m['params'], m['f1'], s=m['size'], c=m['color'], marker=m['marker'],
                   edgecolors=edge_color, linewidths=edge_w, zorder=5)
        
        # Label with throughput note
        label_text = f"{m['short_name']}\nSpeed: {m['badge']}"
        if m['highlight']:
            label_text = f"★ {m['short_name']}\nSpeed: {m['badge']} (Champion)"
            font_weight = 'bold'
            font_color = '#962d22'
            bbox_props = dict(boxstyle='round,pad=0.3', facecolor='#fdedec', edgecolor='#f5b7b1', alpha=0.9, linewidth=1.0)
        elif m['name'] == 'Tier 5 (132k)':
            label_text = f"{m['short_name']}\nSpeed: {m['badge']} (0 FP)"
            font_weight = 'semibold'
            font_color = '#1e8449'
            bbox_props = dict(boxstyle='round,pad=0.25', facecolor='#eafaf1', edgecolor='#a9dfbf', alpha=0.9, linewidth=0.8)
        else:
            font_weight = 'normal'
            font_color = '#2c3e50'
            bbox_props = dict(boxstyle='round,pad=0.2', facecolor='#f8f9fa', edgecolor='#d5dbdb', alpha=0.8, linewidth=0.6)
            
        ax.annotate(
            label_text,
            xy=(m['params'], m['f1']),
            xytext=m['text_offset'],
            textcoords='offset points',
            fontsize=9.2,
            fontweight=font_weight,
            color=font_color,
            ha=m['ha'],
            va=m['va'],
            bbox=bbox_props,
            arrowprops=dict(arrowstyle='-', color='#bdc3c7', linewidth=0.8) if (m['text_offset'][0] < 0 or abs(m['text_offset'][1]) > 10) else None,
            zorder=6
        )
    
    # Configure axes
    ax.set_xscale('log')
    ax.set_xlim(8e4, 7e6)
    ax.set_ylim(0.795, 0.905)
    
    ax.set_title('PoseConv3D Capacity Scaling vs. F1-Score & Throughput', 
                 fontsize=13.5, fontweight='bold', pad=14, color='#1a252f')
    ax.set_xlabel('Model Parameters (Log Scale)', fontsize=11, fontweight='semibold', labelpad=10, color='#2c3e50')
    ax.set_ylabel('F1-Score (Violence Detection)', fontsize=11, fontweight='semibold', labelpad=10, color='#2c3e50')
    
    # Legend
    legend = ax.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#d0d7de', fontsize=9.5)
    legend.get_frame().set_alpha(0.95)
    
    # Spines styling
    for spine in ax.spines.values():
        spine.set_color('#bdc3c7')
        spine.set_linewidth(1.0)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='#ffffff', edgecolor='none')
    plt.close()
    print(f"Pareto frontier saved successfully to {output_path}")

if __name__ == "__main__":
    generate_pareto_chart()
