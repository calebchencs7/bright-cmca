# Nano Banana 绘图提示词 — BRIGHT-CMCA 项目三图

> 适用模型：Google Nano Banana (Gemini image)。建议生成比例 16:9，分辨率拉到最高。
> 三张图共用一套视觉规范，保证风格统一、达到 CVPR / TPAMI / ISPRS / RSE 顶刊插图水准。
> 使用方式：直接把对应 PROMPT 整段粘贴给 Nano Banana。若文字渲染有错别字，可追加一句
> "Re-render with all text labels spelled exactly as written, crisp and legible."

---

## 共用视觉规范（已写进每个 prompt，无需单独发）

- Flat vector scientific-diagram style, clean white background, generous whitespace.
- Modern sans-serif typography (Helvetica / Arial), labels crisp and perfectly legible, no gibberish text.
- Rounded-rectangle blocks with thin dark outlines and soft pastel fills; subtle, very light drop shadows only.
- Color coding — Optical branch: cool blue (#3A7BD5); SAR branch: warm amber/orange (#F2994A); Attention / fusion: violet (#7B61FF); Loss: teal (#1AAE9F); Damage colormap output: background = light gray, intact = green (#2E9E5B), damaged = amber (#F2B705), destroyed = red (#D7263D).
- Directional arrows clean and uniform; a small legend box; thin module-grouping frames with rounded corners.
- No photo-realistic clutter, no 3D bevels, no neon glow. Publication-grade, minimalist, high resolution.

---

## 图 1 — 技术路线图 / Overall Framework（横版 16:9）

**PROMPT:**

```
A clean, publication-quality technical roadmap figure for a remote-sensing deep-learning paper, drawn as a flat vector scientific diagram on a pure white background, 16:9 landscape, modern Helvetica sans-serif labels, crisp and perfectly legible text, soft pastel color blocks with thin dark outlines and very subtle shadows, in the style of a CVPR / ISPRS top-journal pipeline figure. Horizontal left-to-right flow organized in four labeled stages with thin rounded grouping frames.

STAGE 1 — "Input: BRIGHT Multimodal Dataset" (left): two stacked image icons, the top one a blue-tinted "Pre-event Optical (VHR RGB)" tile, the bottom one an amber-tinted "Post-event SAR" tile, with a small caption "14 disaster events, globally distributed, 4 damage levels".

STAGE 2 — "Motivation / Challenges" (a small vertical list box in violet): four bullet chips reading "Damaged-class IoU very low (35-48%)", "Optical+SAR fusion gain only +1%", "Poor cross-event generalization", "SAR speckle & ~1px misalignment".

STAGE 3 — "Three Contributions" (center, the visual core): three parallel horizontal lanes, each a rounded block with an icon:
  Lane 1 (blue) "CMCA: Cross-Modal Change Attention — architecture-level optical/SAR fusion".
  Lane 2 (violet) "DPCL: Damage Prototype Contrastive Learning — representation-level class separability".
  Lane 3 (amber) "DACutMix-Training: Damage-Aware CutMix — data-level cross-event generalization & damage-imbalance".
  These three lanes feed into a shared block "Dual-branch UNet backbone".

STAGE 4 — "Training & Output" (right): a loss box (teal) reading "Class-weighted CE + 0.75 x Lovasz-softmax + lambda x DPCL InfoNCE", an arrow to an output map labeled "4-class Damage Map: Background / Intact / Damaged / Destroyed" rendered as a small segmentation tile using a gray/green/amber/red colormap, then an evaluation box "Per-class IoU & mIoU | Standard split + Cross-event transfer | 2x2 ablation".

Add a small legend in a corner mapping the four output colors to background/intact/damaged/destroyed. Uniform clean arrows connecting all stages left to right. Minimalist, no clutter, no 3D, high resolution, balanced composition.
```

---

## 图 2 — CMCA 模型结构图（横版 16:9）

**PROMPT:**

```
A clean, publication-quality neural-network architecture diagram for a dual-branch UNet with a cross-modal attention module, flat vector scientific style on a pure white background, 16:9 landscape, modern Helvetica sans-serif labels with crisp legible text, soft pastel blocks with thin dark outlines and very subtle shadows, in the style of a TPAMI / CVPR architecture figure. A symmetric U-shaped layout.

INPUTS (left): top input "Pre-event Optical (3 channels)" tinted cool blue; bottom input "Post-event SAR (3 channels)" tinted warm amber. The two go into two PARALLEL encoders drawn as descending stacks of convolutional blocks, each block a rounded rectangle labeled with channel counts 64, 128, 256, 512, 512, with downward max-pool arrows between them. Top stack = blue "Optical Encoder", bottom stack = amber "SAR Encoder".

BOTTLENECK CENTER — the highlighted core module in a violet rounded frame titled "CMCA: Cross-Modal Change Attention". Inside it show the data flow: optical bottleneck feature -> "Q (1x1 conv)"; SAR bottleneck feature -> a small "Spatial Reduction (sr=2)" block -> "K, V (1x1 conv)"; then "Scaled Dot-Product Cross-Attention (softmax)" producing "Aligned SAR"; then a concat of [Aligned SAR ; Optical] -> "Change Projection (1x1 conv + BN + ReLU)" -> output "Change feature". A short caption: "Optical queries SAR to localize structural change".

FUSION: a block "Bottleneck Fusion: concat[Optical, SAR, Change] -> 1x1 conv -> 1024".

DECODER (right, ascending): a single stack of up-conv blocks 512 -> 256 -> 128 -> 64 with upward transpose-conv arrows, mirroring the encoders. SKIP CONNECTIONS: at each resolution, draw curved arrows from BOTH the optical and SAR encoder blocks merging through a small "1x1 skip-fusion" node into the corresponding decoder block. Final block -> "1x1 conv" -> output "4-class Damage Map" shown as a small gray/green/amber/red segmentation tile.

Use blue for the optical pathway, amber for the SAR pathway, violet for the attention/fusion, and a clean tensor-shape annotation style. Small legend for the color coding. Symmetric, elegant, minimalist, no 3D, high resolution.
```

---

## 图 3 — DACutMix-Training 模型图（横版 16:9）

**PROMPT:**

```
A clean, publication-quality data-augmentation / training-strategy diagram for a building-damage segmentation paper, flat vector scientific style, pure white background, 16:9 landscape, modern Helvetica sans-serif labels with crisp legible text, soft pastel blocks, thin dark outlines, very subtle shadows, in the style of an ISPRS / RSE methodology figure. Title at top: "DACutMix-Training: Damage-Aware CutMix". The figure has TWO clearly separated horizontal components inside thin rounded frames.

COMPONENT A (left) — "Damage-Balanced Sampler". Show a small bar chart of class frequency where "Damaged" and "Destroyed" bars are tiny next to a huge "Background/Intact" bar, captioned "Severe damage-class imbalance". An arrow to a formula chip: "weight = 1 + boost x min(damage_fraction / target, 1)". Then a row of dataset tiles, with damage-rich tiles (red/amber speckled) drawn LARGER / duplicated to show "WeightedRandomSampler oversamples damage-rich tiles". Use amber/red accents.

COMPONENT B (right) — "Event-aware, Damage-aware CutMix". Show a "Donor pool" as a row of small tiles each tagged with a different event name chip (e.g. "Event A", "Event B", "Event C"), in different muted colors. A target tile labeled "Target tile (Event A)" with three aligned layers stacked and labeled "Optical / SAR / Label". From a DIFFERENT-event donor tile, a dashed bounding box marked "damage patch (class 2/3)" is cut and pasted with an arrow onto the same location of all three aligned target layers, producing a "Cross-event hybrid sample" with more damage pixels. Caption: "Paste a damage-rich patch from another event onto optical+SAR+label jointly".

Both components feed downward with arrows into a single bottom block: "Balanced training batches -> UNet-CMCA". Add a compact legend mapping the damage colormap (background gray / intact green / damaged amber / destroyed red). Balanced two-panel composition, uniform arrows, minimalist, no 3D, high resolution.
```

---

## 小贴士（提升出图质量）

1. **文字易错**：架构图里通道数、公式最容易出错。出图后逐一核对 `64/128/256/512`、`sr=2`、`1x1 conv`、`Lovasz`、公式 `weight = 1 + boost x min(...)`，错了就追加 "fix the text in block X to read exactly: …"。
2. **风格统一**：三图分开生成时，可在第 2、3 个 prompt 前加一句 "Match the exact visual style, color palette and typography of the previous figure."
3. **配色一致性**：始终保持 Optical=蓝、SAR=琥珀、Attention/Fusion=紫、Loss=青、损伤等级 gray/green/amber/red 四色，跨三张图一致，审稿人一眼能对应。
4. **要矢量感**：如果出来太"插画风"，追加 "more like a precise vector schematic, thinner lines, less illustration, more diagrammatic"。
5. **导出**：Nano Banana 出的是位图；若投稿需要矢量，可拿生成图作为版式参考，再用 PowerPoint / draw.io / Inkscape 重绘成矢量。
```
