"""Draw the exact skip and bottleneck connections implemented by UNetCMCA."""

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


def box(ax, x, y, w, h, label, fc, ec, size=10.5, weight="normal"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.03,rounding_size=0.08",
                       linewidth=1.1, facecolor=fc, edgecolor=ec)
    ax.add_patch(p)
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", color=INK,
            fontsize=size, fontweight=weight, linespacing=1.05)


def arrow(ax, a, b, color=INK, dashed=False, lw=1.2, ms=11):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=ms,
                                linewidth=lw, color=color,
                                linestyle="--" if dashed else "-",
                                shrinkA=2, shrinkB=2))


fig, ax = plt.subplots(figsize=(14.6, 7.1), dpi=220)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 14.6)
ax.set_ylim(0, 7.1)
ax.axis("off")

ax.text(7.3, 6.80, "UNetCMCA connections in the current implementation",
        ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
ax.text(0.35, 6.42,
        "Encoder levels are E1–E5. E5 is the bottleneck; paired skips come from E1–E4.",
        ha="left", va="center", fontsize=10.5, color=INK)

headers = [(0.38, "PAIRED ENCODER FEATURES"), (3.35, "SKIP FUSION"),
           (6.15, "DECODER CONCATENATION"), (10.32, "DECODER STAGE")]
for x, label in headers:
    ax.text(x, 5.98, label, ha="left", va="center", fontsize=8.5,
            fontweight="bold", color=GRAY_E)

rows = [
    (5.05, "E4", "D4", "skip_fuse4", "upconv4(neck)"),
    (4.10, "E3", "D3", "skip_fuse3", "upconv3(D4)"),
    (3.15, "E2", "D2", "skip_fuse2", "upconv2(D3)"),
    (2.20, "E1", "D1", "skip_fuse1", "upconv1(D2)"),
]
for y, enc, dec, fuse, up in rows:
    box(ax, 0.38, y, 1.16, 0.54, f"Optical {enc}", BLUE, BLUE_E, size=10)
    ax.text(1.70, y+0.27, "+", ha="center", va="center", fontsize=18, color=GRAY_E)
    box(ax, 1.84, y, 1.10, 0.54, f"SAR {enc}", ORANGE, ORANGE_E, size=10)
    arrow(ax, (2.98, y+0.27), (3.35, y+0.27), color=GRAY_E)
    box(ax, 3.39, y, 2.16, 0.54, f"{fuse}\nconcat + 1×1 Conv", GRAY, GRAY_E,
        size=9.6, weight="bold")
    arrow(ax, (5.59, y+0.27), (6.04, y+0.27), color=GRAY_E)
    box(ax, 6.08, y, 2.05, 0.54, f"concat with\n{up}", "#FAFBFC", GRAY_E, size=9.3)
    arrow(ax, (8.17, y+0.27), (9.00, y+0.27), color=PURPLE_E)
    box(ax, 9.04, y, 1.10, 0.54, dec, PURPLE, PURPLE_E, size=12, weight="bold")
    ax.text(10.35, y+0.27, f"{dec} output", ha="left", va="center",
            fontsize=9.5, color=INK)

# Bottleneck path from E5 and its relation to D4.
ax.text(0.38, 1.57, "BOTTLENECK MAIN PATH", ha="left", va="center",
        fontsize=8.5, fontweight="bold", color=GREEN_E)
box(ax, 0.38, 0.76, 1.92, 0.58, "Optical E5 + SAR E5", BLUE, BLUE_E,
    size=10, weight="bold")
arrow(ax, (2.34, 1.05), (2.77, 1.05), color=GREEN_E)
box(ax, 2.81, 0.76, 1.38, 0.58, "CMCA", GREEN, GREEN_E,
    size=11, weight="bold")
arrow(ax, (4.23, 1.05), (4.65, 1.05), color=GREEN_E)
box(ax, 4.69, 0.76, 2.72, 0.58,
    "cat([E5 optical, E5 SAR, CMCA output])\n1×1 Conv + BN + ReLU",
    GRAY, GRAY_E, size=9.1)
arrow(ax, (7.45, 1.05), (8.14, 1.05), color=PURPLE_E)
box(ax, 8.18, 0.76, 1.15, 0.58, "neck", PURPLE, PURPLE_E,
    size=11, weight="bold")
arrow(ax, (9.37, 1.05), (10.03, 1.05), color=PURPLE_E)
box(ax, 10.07, 0.76, 1.47, 0.58, "upconv4(neck)", PURPLE, PURPLE_E, size=9.6)
arrow(ax, (11.58, 1.05), (12.13, 1.05), color=PURPLE_E)
box(ax, 12.17, 0.76, 1.14, 0.58, "D4", PURPLE, PURPLE_E,
    size=11, weight="bold")
ax.text(12.74, 0.48, "D4 also receives the E4 paired skip above",
        ha="center", va="center", fontsize=8.5, color=GRAY_E)

ax.text(7.3, 0.18, "Decoder flow: D4 → D3 → D2 → D1 → final 1×1 prediction layer",
        ha="center", va="center", fontsize=10, color=INK)

fig.subplots_adjust(left=0.012, right=0.99, top=0.98, bottom=0.035)
fig.savefig(OUT / "UNetCMCA_exact_skip_connections.png", dpi=300,
            facecolor="white", bbox_inches="tight", pad_inches=0.06)
fig.savefig(OUT / "UNetCMCA_exact_skip_connections.svg",
            facecolor="white", bbox_inches="tight", pad_inches=0.06)
plt.close(fig)
