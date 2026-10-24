"""Draw a compact explanatory diagram for paired encoder-decoder skips."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT = Path(__file__).resolve().parents[1] / "images"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, BLUE_E = "#DCEEF7", "#4C94C6"
ORANGE, ORANGE_E = "#FBE4CC", "#D78028"
GREEN, GREEN_E = "#DCEFD9", "#4F9D69"
PURPLE, PURPLE_E = "#E9DFF3", "#8D70AD"
GRAY, GRAY_E = "#F1F3F5", "#737B83"
INK = "#27313A"


def box(ax, x, y, w, h, label, fc, ec, size=11, weight="normal"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.09",
        linewidth=1.15, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=size, color=INK, fontweight=weight, linespacing=1.1)


def arrow(ax, start, end, color=INK, dashed=False, lw=1.25, ms=12,
          connectionstyle="arc3"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
        color=color, linestyle="--" if dashed else "-",
        connectionstyle=connectionstyle, shrinkA=2, shrinkB=2,
    ))


fig, ax = plt.subplots(figsize=(13.2, 5.4), dpi=220)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 13.2)
ax.set_ylim(0, 5.4)
ax.axis("off")

ax.text(6.6, 5.08, "Paired encoder features feed the matching decoder stage",
        ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
ax.text(0.36, 4.66, "OPTICAL", ha="left", va="center", fontsize=9,
        fontweight="bold", color=BLUE_E)
ax.text(1.72, 4.66, "SAR", ha="left", va="center", fontsize=9,
        fontweight="bold", color=ORANGE_E)
ax.text(4.12, 4.66, "PAIR FUSION", ha="left", va="center", fontsize=9,
        fontweight="bold", color=GRAY_E)
ax.text(7.45, 4.66, "PROJECTED SKIP", ha="left", va="center", fontsize=9,
        fontweight="bold", color=GRAY_E)
ax.text(10.12, 4.66, "DECODER", ha="left", va="center", fontsize=9,
        fontweight="bold", color=PURPLE_E)

rows = [
    (3.68, "E3", "D3"),
    (2.55, "E2", "D2"),
    (1.42, "E1", "D1"),
]
for y, enc, dec in rows:
    box(ax, 0.35, y, 1.15, 0.62, f"Optical {enc}", BLUE, BLUE_E, size=11.5)
    ax.text(1.64, y + 0.31, "+", ha="center", va="center", fontsize=19, color=GRAY_E)
    box(ax, 1.82, y, 1.05, 0.62, f"SAR {enc}", ORANGE, ORANGE_E, size=11.5)
    arrow(ax, (2.91, y + 0.31), (3.42, y + 0.31), color=GRAY_E)
    box(ax, 3.46, y, 2.18, 0.62, "Concat + 1×1 projection", GRAY, GRAY_E,
        size=10.5, weight="bold")
    arrow(ax, (5.70, y + 0.31), (8.98, y + 0.31), color=GRAY_E,
          dashed=True, lw=1.45, ms=13)
    ax.text(7.32, y + 0.49, f"{enc} pair → {dec}", ha="center", va="bottom",
            fontsize=9, color=GRAY_E)
    box(ax, 9.08, y, 1.08, 0.62, dec, PURPLE, PURPLE_E, size=14, weight="bold")

# A separate bottleneck note prevents the E4 route from being mistaken for a skip.
box(ax, 0.35, 0.27, 2.10, 0.62, "Optical E4 + SAR E4", BLUE, BLUE_E,
    size=10.2, weight="bold")
arrow(ax, (2.50, 0.58), (3.08, 0.58), color=GREEN_E)
box(ax, 3.12, 0.27, 1.40, 0.62, "CMCA", GREEN, GREEN_E, size=12, weight="bold")
arrow(ax, (4.58, 0.58), (5.16, 0.58), color=GREEN_E)
box(ax, 5.20, 0.27, 2.25, 0.62, "Backbone-specific adapter", GRAY, GRAY_E,
    size=10.2)
arrow(ax, (7.51, 0.58), (8.98, 0.58), color=PURPLE_E)
box(ax, 9.08, 0.27, 1.08, 0.62, "D3 input", PURPLE, PURPLE_E,
    size=10.8, weight="bold")
ax.text(10.38, 0.58, "Main path (not a skip)", ha="left", va="center",
        fontsize=10.5, color=INK)

ax.text(11.65, 3.92, "Decoder flow", ha="center", va="center",
        fontsize=10, fontweight="bold", color=PURPLE_E)
ax.text(11.65, 3.42, "D3  →  D2  →  D1", ha="center", va="center",
        fontsize=11, color=INK)
ax.text(11.65, 2.70, "Each decoder stage receives:\nmain decoder input + its paired skip",
        ha="center", va="center", fontsize=9.5, color=INK, linespacing=1.35)

ax.text(6.6, 0.03,
        "Skip mapping: (Optical E3 + SAR E3) → D3;  (Optical E2 + SAR E2) → D2;  (Optical E1 + SAR E1) → D1",
        ha="center", va="bottom", fontsize=9, color=GRAY_E)

fig.subplots_adjust(left=0.012, right=0.99, top=0.98, bottom=0.03)
fig.savefig(OUT / "E1-E4_to_D3-D1_skip_mapping.png", dpi=300,
            facecolor="white", bbox_inches="tight", pad_inches=0.06)
fig.savefig(OUT / "E1-E4_to_D3-D1_skip_mapping.svg",
            facecolor="white", bbox_inches="tight", pad_inches=0.06)
plt.close(fig)
