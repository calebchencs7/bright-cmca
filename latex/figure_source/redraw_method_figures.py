#!/usr/bin/env python3
"""Render the three method diagrams used by the TGRS manuscript.

The artwork is intentionally generated from vector primitives.  Each figure is
exported both as a high-resolution PNG (the filenames already referenced by
main.tex) and as an editable/vector PDF sibling for final submission.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)


ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = ROOT / "latex" / "images"


# Low-saturation, print-friendly colors shared by all three figures.
INK = "#20252B"
MID = "#687078"
LIGHT_LINE = "#AEB6BE"
PANEL = "#FAFBFC"
OPT = "#CFE6F6"
OPT_DARK = "#4C86B5"
SAR = "#F6D9BC"
SAR_DARK = "#C77932"
CMCA = "#D9EEDC"
CMCA_DARK = "#4F8E63"
DEC = "#E8DDF0"
DEC_DARK = "#86659A"
AUG = "#F8EDBC"
AUG_DARK = "#B18A18"
NEUTRAL = "#EEF1F3"
BG_CLASS = "#D9D9D9"
INTACT = "#54B979"
DAMAGED = "#E6AE3C"
DESTROYED = "#D85A52"


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 7.2,
        "axes.linewidth": 0.7,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }
)


def canvas(width: float, height: float):
    fig = plt.figure(figsize=(width, height), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    return fig, ax


def add_box(
    ax,
    x,
    y,
    w,
    h,
    text="",
    *,
    fc="white",
    ec=INK,
    lw=0.75,
    radius=0.012,
    fontsize=7.0,
    weight="normal",
    color=INK,
    ha="center",
    va="center",
    zorder=3,
    linestyle="-",
    pad=0.006,
    linespacing=1.05,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=linestyle,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if text:
        tx = x + w / 2 if ha == "center" else (x + 0.012 if ha == "left" else x + w - 0.012)
        ax.text(
            tx,
            y + h / 2,
            text,
            ha=ha,
            va=va,
            fontsize=fontsize,
            fontweight=weight,
            color=color,
            linespacing=linespacing,
            zorder=zorder + 1,
        )
    return patch


def add_panel(ax, x, y, w, h, title, *, ec=LIGHT_LINE, dashed=True, fontsize=8.0):
    add_box(
        ax,
        x,
        y,
        w,
        h,
        fc=PANEL,
        ec=ec,
        lw=0.75,
        radius=0.014,
        linestyle=(0, (4, 2)) if dashed else "-",
        pad=0.0,
        zorder=0,
    )
    ax.text(
        x + 0.012,
        y + h - 0.018,
        title,
        ha="left",
        va="top",
        fontsize=fontsize,
        fontweight="bold",
        color=INK,
        zorder=5,
    )


def add_arrow(
    ax,
    start,
    end,
    *,
    color=INK,
    lw=0.8,
    style="-|>",
    mutation=7,
    connectionstyle="arc3,rad=0",
    linestyle="-",
    zorder=2,
):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        linestyle=linestyle,
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=zorder,
    )
    ax.add_patch(arr)
    return arr


def add_route(ax, points, *, color=INK, lw=0.72, linestyle="-", mutation=6.5, zorder=2):
    """Draw an orthogonal route with an arrowhead on the final segment."""
    for p0, p1 in zip(points[:-2], points[1:-1]):
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=color,
            lw=lw,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=zorder,
        )
    add_arrow(
        ax,
        points[-2],
        points[-1],
        color=color,
        lw=lw,
        mutation=mutation,
        linestyle=linestyle,
        zorder=zorder,
    )


def add_number(ax, x, y, value, *, fc=INK):
    c = Circle((x, y), 0.014, facecolor=fc, edgecolor="none", zorder=6)
    ax.add_patch(c)
    ax.text(x, y - 0.001, str(value), ha="center", va="center", color="white", fontsize=6.3, fontweight="bold", zorder=7)


def add_feature_stack(ax, x, y, w, h, *, fc, ec=INK, layers=3, label=None, fontsize=6.4):
    """A restrained shallow stack used for feature tensors."""
    dx, dy = w * 0.10, h * 0.08
    for i in range(layers - 1, -1, -1):
        rect = Rectangle(
            (x + i * dx, y + i * dy),
            w,
            h,
            facecolor=mpl.colors.to_rgba(fc, 0.84 + 0.05 * (layers - 1 - i)),
            edgecolor=ec,
            linewidth=0.65,
            zorder=3 + i,
        )
        ax.add_patch(rect)
    if label:
        ax.text(x + w / 2 + dx, y - 0.015, label, ha="center", va="top", fontsize=fontsize, color=INK, zorder=8)


def add_stage_block(ax, x, y, w, h, label, *, fc, ec=INK, fontsize=6.5):
    add_box(ax, x, y, w, h, label, fc=fc, ec=ec, radius=0.006, pad=0.0018, fontsize=fontsize, lw=0.7)


def add_optical_tile(ax, x, y, w, h, *, border=INK, box=None):
    base = Rectangle((x, y), w, h, facecolor="#71877B", edgecolor=border, linewidth=0.65, zorder=4)
    ax.add_patch(base)
    # Fields, roofs, and one pale road; kept geometric so the thumbnail remains
    # legible at final IEEE column size.
    fields = [
        (0.02, 0.08, 0.30, 0.34, "#677C65"),
        (0.34, 0.04, 0.25, 0.30, "#9A8D69"),
        (0.61, 0.09, 0.34, 0.26, "#6E8064"),
        (0.05, 0.53, 0.34, 0.42, "#566F5B"),
        (0.44, 0.58, 0.50, 0.36, "#8B7B5D"),
    ]
    for fx, fy, fw, fh, c in fields:
        ax.add_patch(Rectangle((x + fx * w, y + fy * h), fw * w, fh * h, facecolor=c, edgecolor="none", zorder=5))
    ax.plot([x + 0.02 * w, x + 0.96 * w], [y + 0.46 * h, y + 0.38 * h], color="#D5D0BE", lw=1.0, zorder=6)
    roofs = [
        (0.18, 0.63, 0.14, 0.11, "#C9B5A4"),
        (0.48, 0.69, 0.17, 0.12, "#D4C7B7"),
        (0.68, 0.52, 0.16, 0.10, "#BFA795"),
        (0.36, 0.18, 0.13, 0.11, "#D0C3B2"),
    ]
    for rx, ry, rw, rh, c in roofs:
        ax.add_patch(Rectangle((x + rx * w, y + ry * h), rw * w, rh * h, facecolor=c, edgecolor="#4F4F4F", linewidth=0.25, zorder=7))
    if box is not None:
        bx, by, bw, bh = box
        ax.add_patch(
            Rectangle(
                (x + bx * w, y + by * h),
                bw * w,
                bh * h,
                facecolor="none",
                edgecolor=AUG_DARK,
                linewidth=1.0,
                linestyle=(0, (3, 1.5)),
                zorder=9,
            )
        )


def add_sar_tile(ax, x, y, w, h, *, border=INK, box=None):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#55595E", edgecolor=border, linewidth=0.65, zorder=4))
    bands = [
        (0.05, 0.10, 0.23, 0.28, "#6D7176"),
        (0.32, 0.08, 0.27, 0.24, "#3E4247"),
        (0.62, 0.06, 0.32, 0.30, "#74787C"),
        (0.04, 0.56, 0.37, 0.38, "#35393D"),
        (0.46, 0.55, 0.48, 0.39, "#656A70"),
    ]
    for fx, fy, fw, fh, c in bands:
        ax.add_patch(Rectangle((x + fx * w, y + fy * h), fw * w, fh * h, facecolor=c, edgecolor="none", zorder=5))
    ax.plot([x + 0.02 * w, x + 0.96 * w], [y + 0.45 * h, y + 0.36 * h], color="#C2C6C9", lw=0.8, zorder=6)
    for rx, ry, rw, rh in [(0.18, 0.65, 0.15, 0.07), (0.50, 0.70, 0.18, 0.08), (0.69, 0.51, 0.13, 0.06), (0.38, 0.20, 0.12, 0.08)]:
        ax.add_patch(Rectangle((x + rx * w, y + ry * h), rw * w, rh * h, facecolor="#E1E3E5", edgecolor="none", zorder=7))
    if box is not None:
        bx, by, bw, bh = box
        ax.add_patch(Rectangle((x + bx * w, y + by * h), bw * w, bh * h, facecolor="none", edgecolor=AUG_DARK, linewidth=1.0, linestyle=(0, (3, 1.5)), zorder=9))


def add_label_tile(ax, x, y, w, h, *, border=INK, box=None, mixed=False):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=BG_CLASS, edgecolor=border, linewidth=0.65, zorder=4))
    shapes = [
        (0.10, 0.63, 0.20, 0.13, INTACT),
        (0.36, 0.70, 0.18, 0.12, INTACT),
        (0.68, 0.58, 0.20, 0.16, INTACT),
        (0.18, 0.22, 0.17, 0.15, INTACT),
        (0.49, 0.31, 0.16, 0.15, DAMAGED),
        (0.70, 0.18, 0.18, 0.18, DESTROYED),
    ]
    if mixed:
        shapes += [(0.41, 0.48, 0.16, 0.15, DAMAGED), (0.59, 0.43, 0.18, 0.17, DESTROYED)]
    for rx, ry, rw, rh, c in shapes:
        ax.add_patch(Rectangle((x + rx * w, y + ry * h), rw * w, rh * h, facecolor=c, edgecolor="none", zorder=6))
    if box is not None:
        bx, by, bw, bh = box
        ax.add_patch(Rectangle((x + bx * w, y + by * h), bw * w, bh * h, facecolor="none", edgecolor=AUG_DARK, linewidth=1.0, linestyle=(0, (3, 1.5)), zorder=9))


def add_triplet(ax, x, y, w, h, *, labels=True, donor=False, mixed=False, vertical=False):
    """Draw registered optical/SAR/label layers."""
    box = (0.42, 0.36, 0.45, 0.38) if donor or mixed else None
    if vertical:
        gap = h * 0.055
        th = (h - 2 * gap) / 3
        items = [(add_optical_tile, "O"), (add_sar_tile, "S"), (add_label_tile, "Y")]
        for idx, (fn, lab) in enumerate(items):
            yy = y + (2 - idx) * (th + gap)
            if fn is add_label_tile:
                fn(ax, x, yy, w, th, box=box, mixed=mixed)
            else:
                fn(ax, x, yy, w, th, box=box if donor or mixed else None)
            if labels:
                ax.text(x - 0.008, yy + th / 2, lab, ha="right", va="center", fontsize=5.8, color=MID)
    else:
        gap = w * 0.055
        tw = (w - 2 * gap) / 3
        fns = [add_optical_tile, add_sar_tile, add_label_tile]
        labs = ["O", "S", "Y"]
        for idx, (fn, lab) in enumerate(zip(fns, labs)):
            xx = x + idx * (tw + gap)
            if fn is add_label_tile:
                fn(ax, xx, y, tw, h, box=box, mixed=mixed)
            else:
                fn(ax, xx, y, tw, h, box=box if donor or mixed else None)
            if labels:
                ax.text(xx + tw / 2, y - 0.014, lab, ha="center", va="top", fontsize=5.8, color=MID)


def add_segmentation_map(ax, x, y, w, h, *, title=None):
    add_label_tile(ax, x, y, w, h, mixed=True)
    if title:
        ax.text(x + w / 2, y + h + 0.014, title, ha="center", va="bottom", fontsize=6.5, fontweight="bold", color=INK)


def add_cylinder(ax, x, y, w, h, text):
    body_h = h * 0.78
    ax.add_patch(Rectangle((x, y + h * 0.10), w, body_h, facecolor=NEUTRAL, edgecolor=INK, linewidth=0.7, zorder=3))
    ax.add_patch(Ellipse((x + w / 2, y + h * 0.88), w, h * 0.22, facecolor="#F7F8F9", edgecolor=INK, linewidth=0.7, zorder=5))
    ax.add_patch(Ellipse((x + w / 2, y + h * 0.10), w, h * 0.22, facecolor=NEUTRAL, edgecolor=INK, linewidth=0.7, zorder=2))
    ax.text(x + w / 2, y + h * 0.48, text, ha="center", va="center", fontsize=6.3, color=INK, linespacing=1.0, zorder=6)


def add_operator(ax, x, y, symbol, *, fc="white", r=0.012, fontsize=7.0):
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=INK, linewidth=0.65, zorder=5))
    ax.text(x, y - 0.0005, symbol, ha="center", va="center", fontsize=fontsize, color=INK, zorder=6)


def save_figure(fig, stem: str, *, dpi=600):
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    png = IMAGE_DIR / f"{stem}.png"
    pdf = IMAGE_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches=None, pad_inches=0, facecolor="white")
    fig.savefig(pdf, bbox_inches=None, pad_inches=0, facecolor="white")
    plt.close(fig)


def roadmap():
    fig, ax = canvas(7.16, 3.05)

    # Three high-level groups.  The overview deliberately avoids repeating the
    # internal mathematics that appears in the two dedicated module figures.
    add_panel(ax, 0.012, 0.105, 0.214, 0.84, "(a) DA-CutMix - training only", ec=AUG_DARK, fontsize=7.9)
    add_panel(ax, 0.240, 0.105, 0.582, 0.84, "(b) Dual-stream network with CMCA", ec=OPT_DARK, fontsize=7.9)
    add_panel(ax, 0.836, 0.105, 0.152, 0.84, "(c) Prediction", ec=DEC_DARK, fontsize=7.9)

    # (a) Target and cross-event donor.
    ax.text(0.061, 0.858, r"Target $\mathcal{T}_i$", ha="center", va="center", fontsize=6.7, fontweight="bold")
    ax.text(0.166, 0.858, r"Donor $\mathcal{T}_j$", ha="center", va="center", fontsize=6.7, fontweight="bold")
    ax.text(0.166, 0.821, r"$e_j\ne e_i$", ha="center", va="center", fontsize=6.5, color=SAR_DARK)
    add_triplet(ax, 0.028, 0.652, 0.066, 0.142, vertical=True)
    add_triplet(ax, 0.134, 0.652, 0.066, 0.142, vertical=True, donor=True)
    add_arrow(ax, (0.064, 0.635), (0.103, 0.555), color=MID, mutation=6)
    add_arrow(ax, (0.167, 0.635), (0.126, 0.555), color=MID, mutation=6)
    add_box(
        ax,
        0.047,
        0.440,
        0.144,
        0.120,
        "Different event\n+ damage-rich box $B$",
        fc=AUG,
        ec=AUG_DARK,
        fontsize=6.7,
        weight="bold",
    )
    ax.text(0.119, 0.406, r"same $B$ in $\mathbf{X}^{o},\mathbf{X}^{s},\mathbf{Y}$", ha="center", va="center", fontsize=6.1, color=MID)
    add_arrow(ax, (0.119, 0.433), (0.119, 0.325), color=AUG_DARK, mutation=6.5)
    add_triplet(ax, 0.071, 0.196, 0.100, 0.106, labels=False, mixed=True)
    ax.text(0.121, 0.163, "Augmented triplet", ha="center", va="center", fontsize=6.4, fontweight="bold")

    # Connection from data-side augmentation to the representative network.
    add_arrow(ax, (0.226, 0.500), (0.253, 0.500), color=INK, mutation=7.5, lw=0.9)

    # (b) Registered modality inputs and modality-specific encoder streams.
    add_optical_tile(ax, 0.257, 0.690, 0.050, 0.112)
    add_sar_tile(ax, 0.257, 0.257, 0.050, 0.112)
    ax.text(0.282, 0.822, "Pre-event\noptical", ha="center", va="bottom", fontsize=6.5, linespacing=0.95)
    ax.text(0.282, 0.237, "Post-event\nSAR", ha="center", va="top", fontsize=6.5, linespacing=0.95)

    enc_x = [0.333, 0.393, 0.453, 0.513]
    widths = [0.035, 0.032, 0.029, 0.026]
    heights = [0.132, 0.115, 0.098, 0.083]
    for idx, (xx, ww, hh) in enumerate(zip(enc_x, widths, heights), 1):
        oy = 0.745 - hh / 2
        sy = 0.313 - hh / 2
        add_feature_stack(ax, xx, oy, ww, hh, fc=OPT, layers=2)
        add_feature_stack(ax, xx, sy, ww, hh, fc=SAR, layers=2)
        ax.text(xx + ww / 2 + ww * 0.10, oy + hh + 0.020, f"E{idx}", ha="center", va="bottom", fontsize=6.1, color=OPT_DARK)
        ax.text(xx + ww / 2 + ww * 0.10, sy - 0.018, f"E{idx}", ha="center", va="top", fontsize=6.1, color=SAR_DARK)
        if idx == 1:
            add_arrow(ax, (0.309, 0.746), (xx - 0.002, 0.746), color=OPT_DARK, mutation=5.5)
            add_arrow(ax, (0.309, 0.313), (xx - 0.002, 0.313), color=SAR_DARK, mutation=5.5)
        else:
            px = enc_x[idx - 2] + widths[idx - 2] * 1.18
            add_arrow(ax, (px, 0.746), (xx - 0.002, 0.746), color=OPT_DARK, mutation=5.5)
            add_arrow(ax, (px, 0.313), (xx - 0.002, 0.313), color=SAR_DARK, mutation=5.5)

    # CMCA is the sole deep cross-modal interaction shown in the overview.
    add_arrow(ax, (0.543, 0.746), (0.570, 0.574), color=OPT_DARK, mutation=6.2)
    add_arrow(ax, (0.543, 0.313), (0.570, 0.458), color=SAR_DARK, mutation=6.2)
    add_box(
        ax,
        0.570,
        0.430,
        0.098,
        0.165,
        "CMCA\n$\mathbf{F}^{o}_{b},\mathbf{F}^{s}_{b}\mapsto\mathbf{F}^{c}_{b}$",
        fc=CMCA,
        ec=CMCA_DARK,
        fontsize=6.8,
        weight="bold",
        linespacing=1.10,
    )
    ax.text(0.619, 0.406, "deep interaction", ha="center", va="top", fontsize=5.9, color=CMCA_DARK)

    add_box(
        ax,
        0.687,
        0.457,
        0.072,
        0.112,
        "Backbone-\nspecific\nadapter",
        fc=NEUTRAL,
        ec=MID,
        fontsize=6.1,
        linespacing=0.92,
    )
    add_arrow(ax, (0.668, 0.513), (0.684, 0.513), color=CMCA_DARK, mutation=6.0)

    # Existing decoder/head.  Paired shallow features enter via projected skip
    # fusion, indicated by thin dashed curves instead of dense duplicated lanes.
    dec_x = [0.770, 0.770, 0.770]
    dec_y = [0.430, 0.552, 0.674]
    dec_w = [0.036, 0.036, 0.036]
    dec_h = [0.078, 0.078, 0.078]
    add_arrow(ax, (0.759, 0.513), (0.767, 0.470), color=DEC_DARK, mutation=5.5)
    for i, (xx, yy, ww, hh) in enumerate(zip(dec_x, dec_y, dec_w, dec_h), 1):
        add_stage_block(ax, xx, yy, ww, hh, f"D{4-i}", fc=DEC, ec=DEC_DARK, fontsize=6.1)
        if i < len(dec_x):
            add_arrow(ax, (xx + ww / 2, yy + hh), (dec_x[i] + dec_w[i] / 2, dec_y[i] - 0.002), color=DEC_DARK, mutation=5.0)
    ax.text(0.788, 0.395, "existing decoder", ha="center", va="top", fontsize=5.9, color=MID)

    # Three paired skip projections.  Each curve starts midway between the
    # optical and SAR features at the corresponding scale.
    skip_targets = [(0.768, 0.469), (0.768, 0.591), (0.768, 0.713)]
    for idx, (sx, target) in enumerate(zip(enc_x[:3], skip_targets), 1):
        midx = sx + widths[idx - 1] * 0.56
        ax.plot([midx, midx], [0.385, 0.665], color=LIGHT_LINE, lw=0.6, linestyle=(0, (2, 1.5)), zorder=1)
        add_arrow(
            ax,
            (midx, 0.525),
            target,
            color=LIGHT_LINE,
            lw=0.62,
            mutation=5.0,
            connectionstyle=f"arc3,rad={-0.18 + idx * 0.02}",
            linestyle=(0, (2, 1.5)),
            zorder=1,
        )
        ax.text(midx, 0.536, rf"$S_{idx}$", ha="center", va="bottom", fontsize=5.3, color=MID)
    ax.text(
        0.435,
        0.535,
        r"paired skips: concat $+\,1\times1$ projection",
        ha="center",
        va="bottom",
        fontsize=5.7,
        color=MID,
        bbox=dict(facecolor=PANEL, edgecolor="none", pad=0.8),
        zorder=6,
    )

    # (c) prediction and supervision.  No fixed loss coefficient is asserted.
    add_arrow(ax, (0.806, 0.713), (0.850, 0.708), color=INK, mutation=7)
    add_segmentation_map(ax, 0.860, 0.625, 0.102, 0.145, title="Damage map")
    add_box(
        ax,
        0.856,
        0.404,
        0.110,
        0.112,
        r"$\mathcal{L}_{dam}$" + "\n" + r"$=\mathcal{L}_{CE}+\lambda_{dam}\mathcal{L}_{Lovasz}$",
        fc=DEC,
        ec=DEC_DARK,
        fontsize=6.15,
        linespacing=1.05,
    )
    add_arrow(ax, (0.911, 0.622), (0.911, 0.521), color=INK, mutation=5.5, linestyle=(0, (3, 2)))
    add_label_tile(ax, 0.877, 0.180, 0.068, 0.102)
    ax.text(0.911, 0.156, "Ground truth", ha="center", va="top", fontsize=6.2)
    add_arrow(ax, (0.911, 0.286), (0.911, 0.400), color=MID, mutation=5.5, linestyle=(0, (3, 2)))

    # Compact visual grammar.  The caption carries the narrative.
    legend_y = 0.044
    legend_items = [(OPT, "Optical"), (SAR, "SAR"), (CMCA, "CMCA"), (DEC, "Decoder"), (AUG, "DA-CutMix")]
    x = 0.243
    for color, label in legend_items:
        ax.add_patch(Rectangle((x, legend_y - 0.012), 0.018, 0.024, facecolor=color, edgecolor=INK, linewidth=0.45, zorder=4))
        ax.text(x + 0.023, legend_y, label, ha="left", va="center", fontsize=5.8, color=INK)
        x += 0.080 if label != "DA-CutMix" else 0.090
    ax.plot([0.682, 0.710], [legend_y, legend_y], color=LIGHT_LINE, lw=0.7, linestyle=(0, (2, 1.5)))
    ax.text(0.716, legend_y, "projected skip / supervision", ha="left", va="center", fontsize=5.8, color=MID)

    save_figure(fig, "Technology Roadmap")


def cmca_figure():
    fig, ax = canvas(3.50, 2.78)

    # Subtle outer boundary matches the module-inset convention common in TGRS.
    add_box(ax, 0.015, 0.038, 0.970, 0.925, fc="white", ec=LIGHT_LINE, lw=0.7, radius=0.012, pad=0.0, zorder=0)

    # Inputs.
    ax.text(0.035, 0.908, "Optical feature", ha="left", va="center", fontsize=6.8, fontweight="bold")
    ax.text(0.035, 0.405, "SAR feature", ha="left", va="center", fontsize=6.8, fontweight="bold")
    add_feature_stack(ax, 0.055, 0.690, 0.095, 0.120, fc=OPT, layers=3)
    add_feature_stack(ax, 0.055, 0.190, 0.095, 0.120, fc=SAR, layers=3)
    ax.text(0.110, 0.655, r"$\mathbf{F}^{o}_{b}$", ha="center", va="top", fontsize=8.0, color=OPT_DARK)
    ax.text(0.110, 0.155, r"$\mathbf{F}^{s}_{b}$", ha="center", va="top", fontsize=8.0, color=SAR_DARK)

    # Optical query lane.
    add_box(ax, 0.205, 0.690, 0.145, 0.112, r"$1\times1$ Conv" + "\n" + r"$\mathbf{Q}$", fc=OPT, ec=OPT_DARK, fontsize=6.7, linespacing=1.0)
    add_arrow(ax, (0.158, 0.752), (0.202, 0.746), color=OPT_DARK, mutation=6)

    # SAR-only spatial reduction and K/V projections.
    add_box(ax, 0.185, 0.192, 0.185, 0.142, fc=SAR, ec=SAR_DARK, fontsize=6.0)
    ax.text(0.278, 0.290, "Spatial reduction", ha="center", va="center", fontsize=5.8, fontweight="bold", color=INK)
    ax.text(0.278, 0.255, "DWConv", ha="center", va="center", fontsize=5.6, color=INK)
    ax.text(0.278, 0.220, r"$(k=s=r)$ + BN", ha="center", va="center", fontsize=5.4, color=INK)
    add_arrow(ax, (0.158, 0.252), (0.189, 0.262), color=SAR_DARK, mutation=6)
    add_box(ax, 0.408, 0.270, 0.128, 0.100, r"$1\times1$ Conv" + "\n" + r"$\mathbf{K}$", fc=SAR, ec=SAR_DARK, fontsize=6.5, linespacing=1.0)
    add_box(ax, 0.408, 0.118, 0.128, 0.100, r"$1\times1$ Conv" + "\n" + r"$\mathbf{V}$", fc=SAR, ec=SAR_DARK, fontsize=6.5, linespacing=1.0)
    add_arrow(ax, (0.370, 0.270), (0.405, 0.320), color=SAR_DARK, mutation=5.5)
    add_arrow(ax, (0.370, 0.252), (0.405, 0.168), color=SAR_DARK, mutation=5.5)

    # Cross-attention is shown as one compact computation, not a detached inset.
    add_box(ax, 0.430, 0.505, 0.255, 0.210, fc=CMCA, ec=CMCA_DARK, fontsize=6.3)
    ax.text(0.558, 0.668, "Multi-head cross-attention", ha="center", va="center", fontsize=6.4, fontweight="bold", color=INK)
    ax.text(0.558, 0.606, r"$\mathbf{A}=\operatorname{softmax}_{N_s}(\mathbf{Q}\mathbf{K}^{\mathsf{T}}/\sqrt{d})$", ha="center", va="center", fontsize=5.8, color=INK)
    ax.text(0.558, 0.548, r"$\widetilde{\mathbf{F}}^{s}_{b}=\mathbf{A}\mathbf{V}$", ha="center", va="center", fontsize=6.1, color=INK)
    add_arrow(ax, (0.350, 0.746), (0.427, 0.642), color=OPT_DARK, mutation=6.0)
    add_route(ax, [(0.536, 0.320), (0.570, 0.320), (0.570, 0.502)], color=SAR_DARK, mutation=5.5)
    add_route(ax, [(0.536, 0.168), (0.606, 0.168), (0.606, 0.502)], color=SAR_DARK, mutation=5.5)

    # Learned comparison with the original optical representation.
    add_operator(ax, 0.737, 0.610, "C", fc=NEUTRAL, r=0.019, fontsize=7.0)
    ax.text(0.737, 0.573, "concat", ha="center", va="top", fontsize=5.7, color=MID)
    add_arrow(ax, (0.685, 0.610), (0.716, 0.610), color=CMCA_DARK, mutation=5.5)
    add_route(ax, [(0.155, 0.825), (0.737, 0.825), (0.737, 0.633)], color=OPT_DARK, mutation=5.5)
    add_box(ax, 0.785, 0.530, 0.150, 0.160, r"$1\times1$ Conv" + "\nBN + ReLU", fc=DEC, ec=DEC_DARK, fontsize=6.6, weight="bold")
    add_arrow(ax, (0.758, 0.610), (0.782, 0.610), color=INK, mutation=5.5)
    add_arrow(ax, (0.860, 0.527), (0.860, 0.397), color=DEC_DARK, mutation=6.0)
    add_feature_stack(ax, 0.802, 0.250, 0.105, 0.118, fc=DEC, ec=INK, layers=3)
    ax.text(0.860, 0.210, r"$\mathbf{F}^{c}_{b}$", ha="center", va="top", fontsize=8.2, color=DEC_DARK)
    ax.text(0.860, 0.158, "change-aware feature", ha="center", va="top", fontsize=6.0, color=INK)

    ax.text(0.500, 0.058, r"$\mathbf{Q}:M\times N\times d;\quad \mathbf{K},\mathbf{V}:M\times N_s\times d;\quad N=h_bw_b;\quad d=C_b/M$", ha="center", va="center", fontsize=5.25, color=MID)

    save_figure(fig, "CMCA")


def dacutmix_figure():
    fig, ax = canvas(3.50, 2.58)

    add_box(ax, 0.015, 0.070, 0.970, 0.890, fc="white", ec=LIGHT_LINE, lw=0.7, radius=0.012, pad=0.0, zorder=0)

    # The training pool branches into a target and a cross-event donor, closely
    # following the compact single-column visual grammar used in TGRS papers.
    add_cylinder(ax, 0.030, 0.400, 0.125, 0.220, "Training\npool")
    add_box(ax, 0.026, 0.680, 0.132, 0.050, "training only", fc=AUG, ec=AUG_DARK, fontsize=5.6, weight="bold", radius=0.010, pad=0.001)

    # Target sample.
    add_box(ax, 0.205, 0.620, 0.245, 0.255, fc="#F8FBFD", ec=OPT_DARK, lw=0.65, radius=0.008, pad=0.002, zorder=1)
    ax.text(0.328, 0.835, r"Target $\mathcal{T}_i$, event $e_i$", ha="center", va="center", fontsize=6.4, fontweight="bold")
    add_triplet(ax, 0.235, 0.678, 0.185, 0.105)

    # Donor sample from a different event; the same candidate B is outlined in
    # all registered layers to prevent any suggestion of modality drift.
    add_box(ax, 0.205, 0.168, 0.245, 0.270, fc="#FEFBF7", ec=SAR_DARK, lw=0.65, radius=0.008, pad=0.002, zorder=1)
    ax.text(0.328, 0.400, r"Donor $\mathcal{T}_j$, $e_j\ne e_i$", ha="center", va="center", fontsize=6.4, fontweight="bold")
    add_triplet(ax, 0.235, 0.235, 0.185, 0.105, donor=True)

    add_arrow(ax, (0.155, 0.555), (0.202, 0.735), color=OPT_DARK, mutation=6.2, connectionstyle="arc3,rad=-0.16")
    add_arrow(ax, (0.155, 0.468), (0.202, 0.305), color=SAR_DARK, mutation=6.2, connectionstyle="arc3,rad=0.16")

    # Damage-aware acceptance test.
    add_box(ax, 0.505, 0.590, 0.270, 0.245, fc=CMCA, ec=CMCA_DARK, fontsize=6.0)
    ax.text(0.640, 0.795, r"Damage-aware box $B$", ha="center", va="center", fontsize=6.4, fontweight="bold")
    ax.text(0.640, 0.718, r"$\sum_{\mathbf{u}\in B}\mathbb{I}[\mathbf{Y}_j(\mathbf{u})\in\{2,3\}]$", ha="center", va="center", fontsize=5.8)
    ax.text(0.640, 0.653, r"$\geq\max(n_{min},\tau|B|)$", ha="center", va="center", fontsize=6.0)
    ax.text(0.640, 0.610, "accept", ha="center", va="center", fontsize=5.4, color=CMCA_DARK, fontweight="bold")
    add_arrow(ax, (0.450, 0.305), (0.552, 0.588), color=SAR_DARK, mutation=6.0, connectionstyle="arc3,rad=-0.10")

    # Joint hard paste with a single shared binary mask.
    add_box(ax, 0.525, 0.225, 0.230, 0.185, fc=AUG, ec=AUG_DARK, fontsize=6.0)
    ax.text(0.640, 0.360, "DA-CutMix", ha="center", va="center", fontsize=6.7, fontweight="bold")
    ax.text(0.640, 0.300, r"same box $B$ pasted into", ha="center", va="center", fontsize=5.8)
    ax.text(0.640, 0.252, r"$\mathbf{X}^{o},\ \mathbf{X}^{s},\ \mathbf{Y}$", ha="center", va="center", fontsize=6.5)
    add_arrow(ax, (0.450, 0.735), (0.570, 0.412), color=OPT_DARK, mutation=6.0, connectionstyle="arc3,rad=0.12")
    add_arrow(ax, (0.640, 0.588), (0.640, 0.414), color=CMCA_DARK, mutation=6.0)

    # Mixed triplet.
    add_box(ax, 0.815, 0.315, 0.150, 0.340, fc="#FFFDF4", ec=AUG_DARK, lw=0.7, radius=0.008, pad=0.002, zorder=1)
    ax.text(0.890, 0.620, r"Mixed $\widetilde{\mathcal{T}}_i$", ha="center", va="center", fontsize=6.5, fontweight="bold")
    add_triplet(ax, 0.852, 0.390, 0.076, 0.175, labels=False, mixed=True, vertical=True)
    add_arrow(ax, (0.755, 0.315), (0.812, 0.458), color=AUG_DARK, mutation=6.5, connectionstyle="arc3,rad=-0.10")

    # Minimal layer legend.
    legend_y = 0.042
    for xx, color, lab in [(0.265, OPT, "O: optical"), (0.435, SAR, "S: SAR"), (0.575, BG_CLASS, "Y: label")]:
        ax.add_patch(Rectangle((xx, legend_y - 0.010), 0.022, 0.020, facecolor=color, edgecolor=INK, linewidth=0.4, zorder=4))
        ax.text(xx + 0.030, legend_y, lab, ha="left", va="center", fontsize=5.7, color=MID)

    save_figure(fig, "DA-CutMix")


def main():
    roadmap()
    cmca_figure()
    dacutmix_figure()
    print(f"Wrote figures to {IMAGE_DIR}")


if __name__ == "__main__":
    main()
