import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

import { buildSlide01 } from "./codex-grid/slide-01.mjs";
import { buildSlide03 } from "./codex-grid/slide-03.mjs";
import { buildSlide05 } from "./codex-grid/slide-05.mjs";
import { buildSlide08 } from "./codex-grid/slide-08.mjs";
import { buildSlide09 } from "./codex-grid/slide-09.mjs";
import { buildSlide10 } from "./codex-grid/slide-10.mjs";
import { buildSlide11 } from "./codex-grid/slide-11.mjs";
import { buildSlide13 } from "./codex-grid/slide-13.mjs";
import { buildSlide14 } from "./codex-grid/slide-14.mjs";
import { buildSlide15 } from "./codex-grid/slide-15.mjs";
import { buildSlide17 } from "./codex-grid/slide-17.mjs";
import { buildSlide18 } from "./codex-grid/slide-18.mjs";
import { buildSlide19 } from "./codex-grid/slide-19.mjs";
import { buildSlide21 } from "./codex-grid/slide-21.mjs";
import { buildSlide26 } from "./codex-grid/slide-26.mjs";

const ROOT = "/Users/haochen/Documents/Development/bright-cmca";
const TMP = path.join(ROOT, "tmp/ai-research-ppt");
const OUT = path.join(ROOT, "AI科研工作流_从Idea到论文_30分钟分享.pptx");
const RENDER_DIR = path.join(TMP, "rendered");

const FONT = "PingFang SC";
const MONO = "SFMono-Regular";
const INK = "#000000";
const MUTED = "#4B5563";
const LIGHT = "#F2F2F2";
const PANEL = "#EDEDED";
const RULE = "#B8BCC4";
const BLUE = "#3D8DFF";
const PALE_BLUE = "#D0EDFA";
const CYAN = "#6DCBF4";
const RED = "#D94C4C";
const GREEN = "#16856B";

const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

function rt(text, size = 22, opts = {}) {
  return {
    runs: [{
      run: text,
      textStyle: {
        fontSize: `${size}px`,
        typeface: opts.typeface || FONT,
        color: opts.color || INK,
        ...(opts.bold ? { bold: true } : {}),
        ...(opts.italic ? { italic: true } : {}),
      },
    }],
    ...(opts.spaceBefore ? { spaceBefore: opts.spaceBefore } : {}),
    ...(opts.spaceAfter ? { spaceAfter: opts.spaceAfter } : {}),
    paragraphStyle: {
      lineSpacingPercent: opts.lineSpacingPercent || 112000,
      ...(opts.alignment ? { alignment: opts.alignment } : {}),
    },
  };
}

function title(text) { return rt(text, 48, { bold: true, lineSpacingPercent: 100000 }); }
function footer(n) { return rt(String(n).padStart(2, "0"), 14, { color: MUTED }); }
function block(titleText, bodyText, bodySize = 22) {
  return {
    titleHere: rt(titleText, 28, { bold: true, spaceAfter: 800 }),
    loremIpsumDolorSitAmetConsecteturAdipiscing: rt(bodyText, bodySize, { color: MUTED }),
  };
}
function blockGoes(titleText, bodyText, bodySize = 22) {
  return {
    titleGoesHere: rt(titleText, 28, { bold: true, spaceAfter: 800 }),
    loremIpsumDolorSitAmetConsecteturAdipiscing: rt(bodyText, bodySize, { color: MUTED }),
  };
}
function topicBlock(topicText, bodyText, extra = "") {
  return {
    topic: rt(topicText, 24, { bold: true, spaceAfter: 500 }),
    loremIpsumDolorSitAmetConsecteturAdipiscing: rt(bodyText, 22, { color: MUTED, spaceAfter: extra ? 500 : 0 }),
    loremIpsumDolorSitAmetConsecteturAdipiscing2: rt(extra, 22, { color: MUTED }),
  };
}

function notes(slide, timing, lines, sources = []) {
  const sourceBlock = sources.length
    ? `\n\n[Sources]\n${sources.map((s) => `- ${s}`).join("\n")}`
    : "";
  slide.speakerNotes.textFrame.setText(
    `建议讲法（${timing}）\n${lines.map((line) => `• ${line}`).join("\n")}${sourceBlock}`,
  );
  slide.speakerNotes.setVisible(true);
}

function addCaseTag(slide, text, left = 42, top = 102) {
  const bar = slide.shapes.add({
    geometry: "rect",
    position: { left, top: top + 2, width: 9, height: 28 },
    fill: BLUE,
    line: { style: "solid", fill: BLUE, width: 0 },
  });
  const label = slide.shapes.add({
    geometry: "textbox",
    position: { left: left + 18, top, width: 440, height: 34 },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  label.text = text;
  label.text.style = { fontSize: 18, bold: true, color: BLUE, typeface: FONT };
  bar.sendToBack();
}

function addTextBox(slide, text, position, opts = {}) {
  const box = slide.shapes.add({
    geometry: opts.geometry || "textbox",
    position,
    fill: opts.fill || "none",
    line: opts.line || { style: "solid", fill: "none", width: 0 },
    ...(opts.borderRadius ? { borderRadius: opts.borderRadius } : {}),
    ...(opts.shadow ? { shadow: opts.shadow } : {}),
  });
  box.text = text;
  box.text.style = {
    fontSize: opts.fontSize || 22,
    typeface: opts.typeface || FONT,
    color: opts.color || INK,
    bold: opts.bold || false,
    alignment: opts.alignment || "left",
    verticalAlignment: opts.verticalAlignment || "top",
    ...(opts.autoFit ? { autoFit: opts.autoFit } : {}),
    ...(opts.insets ? { insets: opts.insets } : {}),
  };
  return box;
}

function addPromptBox(slide, text, position) {
  const surface = slide.shapes.add({
    geometry: "roundRect",
    position,
    fill: "#F7F8FA",
    line: { style: "solid", fill: RULE, width: 1 },
    borderRadius: 16,
  });
  surface.text = text;
  surface.text.style = {
    fontSize: 18,
    typeface: FONT,
    color: "#111827",
    verticalAlignment: "top",
    insets: { top: 18, right: 18, bottom: 18, left: 18 },
  };
  return surface;
}

async function addImage(slide, file, position, opts = {}) {
  const bytes = new Uint8Array(await fs.readFile(file));
  const contentType = file.toLowerCase().endsWith(".jpg") || file.toLowerCase().endsWith(".jpeg")
    ? "image/jpeg" : "image/png";
  if (opts.frame !== false) {
    slide.shapes.add({
      geometry: "roundRect",
      position,
      fill: opts.backing || "#F7F8FA",
      line: { style: "solid", fill: opts.lineColor || RULE, width: 1 },
      borderRadius: opts.radius || 14,
    });
  }
  return slide.images.add({
    blob: bytes,
    contentType,
    alt: opts.alt || path.basename(file),
    fit: opts.fit || "contain",
    position,
    geometry: "roundRect",
    borderRadius: opts.radius || 14,
    ...(opts.crop ? { crop: opts.crop } : {}),
  });
}

function addCodeBox(slide, heading, code, position, accent = BLUE) {
  slide.shapes.add({
    geometry: "roundRect",
    position,
    fill: "#111827",
    line: { style: "solid", fill: "#111827", width: 1 },
    borderRadius: 14,
  });
  addTextBox(slide, heading, { left: position.left + 18, top: position.top + 14, width: position.width - 36, height: 28 }, {
    fontSize: 20, bold: true, color: accent,
  });
  addTextBox(slide, code, { left: position.left + 18, top: position.top + 50, width: position.width - 36, height: position.height - 66 }, {
    fontSize: 16, typeface: MONO, color: "#F9FAFB", autoFit: "shrinkText",
  });
}

function coverExistingRegion(slide, position, fill = "#FFFFFF") {
  return slide.shapes.add({
    geometry: "rect",
    position,
    fill,
    line: { style: "solid", fill, width: 0 },
  });
}

// 01 — Title
{
  const slide = buildSlide01(presentation, {
    title: rt("科研分享 · 真实项目拆解", 24, { bold: true, color: BLUE }),
    title2: rt("我如何用 AI\n完成科研闭环", 82, { bold: true, lineSpacingPercent: 92000 }),
    title3: rt("从 idea、文献与实验，到画图和论文写作\nHao Chen", 28, { color: MUTED, lineSpacingPercent: 115000 }),
  });
  slide.shapes.add({ geometry: "rect", position: { left: 1038, top: 184, width: 200, height: 12 }, fill: BLUE, line: { style: "solid", fill: BLUE, width: 0 } });
  slide.shapes.add({ geometry: "rect", position: { left: 1038, top: 211, width: 126, height: 12 }, fill: CYAN, line: { style: "solid", fill: CYAN, width: 0 } });
  notes(slide, "0:30", [
    "开场先说明：这不是一份 AI 工具排行榜，而是一套在真实科研项目里反复使用的工作流。",
    "今天用一个贯穿案例——多模态建筑损伤评估——把每一步拆开，重点展示输入、输出和人工检查。",
  ]);
}

// 02 — Tool categories and human ownership
{
  const slide = buildSlide09(presentation, {
    title: title("工具很多，但我只把它们分成三类助手"),
    body1: topicBlock(
      "示例组合（按你自己的账号替换即可）",
      "检索/阅读：Elicit、Semantic Scholar、NotebookLM  ·  推理/写作：ChatGPT、Claude、DeepSeek  ·  编程/自动化：Codex、GitHub Copilot",
      "它们加速搜索、比较、生成和检查；问题定义、证据判断、最终责任仍在研究者。",
    ),
    body2: block("检索与阅读", "找入口、整理证据、对照多篇论文；不直接相信摘要式结论。", 20),
    body3: block("推理与写作", "扩展假设、重组论证、暴露逻辑缺口；不让模型替我下科学结论。", 20),
    body4: block("编程与验证", "定位接口、生成最小补丁、补测试；每个改动都要能运行、能复现。", 20),
    footer1: footer(2),
  });
  slide.shapes.add({ geometry: "rect", position: { left: 41, top: 620, width: 1198, height: 10 }, fill: BLUE, line: { style: "solid", fill: BLUE, width: 0 } });
  notes(slide, "1:20", [
    "介绍工具时不要逐个演示界面，先讲三种能力：找资料、想/写、写代码/验证。这样听众换任何工具都能复用。",
    "强调三个不能外包的决定：研究问题是否值得做、证据是否足够、论文中的最终表述是否负责。",
    "可以现场问听众：你们最卡的是哪一类？后面每一类都会有案例。",
  ]);
}

// 03 — Agenda / workflow
{
  const slide = buildSlide03(presentation, {
    title: title("30 分钟看完一条可复用的 AI 科研工作流"),
    body1: "",
    footer1: footer(3),
  });
  coverExistingRegion(slide, { left: 35, top: 205, width: 1210, height: 432 });
  const values = [
    ["01", "找到 idea：把失败现象变成可证伪假设"],
    ["02", "文献调研：从“搜论文”升级为证据矩阵"],
    ["03", "代码与实验：最小改动、清晰对照、可复现"],
    ["04", "科研画图：先写结论，再选择图表"],
    ["05", "论文写作：主张—证据—解释，逐句核验"],
  ];
  const table = slide.tables.add({ rows: 5, columns: 2, left: 41, top: 218, width: 1197, height: 392, columnWidths: [110, 1087], values });
  table.borders.assign({ style: "solid", fill: RULE, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 5, columnCount: 1 }).assign({ fill: "#F5F9FF", textStyle: { fontSize: 22, bold: true, color: BLUE, typeface: FONT }, anchor: "middle" });
  table.cells.block({ row: 0, column: 1, rowCount: 5, columnCount: 1 }).assign({ fill: "#FFFFFF", textStyle: { fontSize: 24, color: INK, typeface: FONT }, anchor: "middle" });
  notes(slide, "0:50", [
    "用这一页建立预期：不是五个互相独立的技巧，而是一条闭环；前一步的产物就是后一步的输入。",
    "说明案例页会给可复制提示词，听众不需要预先懂模型结构。",
  ]);
}

// 04 — Case background
{
  const slide = buildSlide08(presentation, {
    title: title("贯穿案例｜建筑损伤评估"),
    body1: block("为什么难？", "灾后光学影像常受云、烟、雨和夜间影响；SAR 能全天候观测，但它与光学“长得完全不一样”。\n\n任务：用灾前光学 + 灾后 SAR，分出背景、完好、受损、毁坏建筑。", 22),
    footer1: footer(4),
  });
  await addImage(slide, path.join(ROOT, "figure/overall.jpg"), { left: 658, top: 120, width: 581, height: 510 }, { alt: "BRIGHT dataset global overview and samples", fit: "contain" });
  addCaseTag(slide, "贯穿案例 · BRIGHT 多模态遥感", 42, 145);
  notes(slide, "1:10", [
    "先用生活语言解释：光学像照片，SAR 像用微波“摸”表面；两者互补，但像素外观差异很大。",
    "指出四分类里，真正重要的“受损/毁坏”像素又最少，因此同时存在跨模态融合和类别不平衡。",
    "这页只讲任务，不讲方法，为后面的 idea 形成制造张力。",
  ], [
    `${ROOT}/README.md`,
    `${ROOT}/figure/overall.jpg`,
  ]);
}

// 05 — Two bottlenecks
{
  const slide = buildSlide05(presentation, {
    title: title("真正的研究起点，是两个反复失败的现象"),
    body1: block("01  融合没有“比较”", "简单拼接让模型看见两种模态，却没有明确告诉它：应该拿灾前建筑结构去哪里寻找灾后变化。\n\n症状：加入 SAR 后提升有限，跨传感器差异可能盖过真实损伤。", 22),
    body2: block("02  稀有类别学不够", "背景与完好建筑占多数；受损、毁坏样本少，而且灾种之间差异大。\n\n症状：整体准确率很好看，但重要类别和新灾害事件表现不稳。", 22),
    footer1: footer(5),
  });
  slide.shapes.add({ geometry: "rect", position: { left: 41, top: 594, width: 581, height: 8 }, fill: BLUE, line: { style: "solid", fill: BLUE, width: 0 } });
  slide.shapes.add({ geometry: "rect", position: { left: 657, top: 594, width: 581, height: 8 }, fill: CYAN, line: { style: "solid", fill: CYAN, width: 0 } });
  notes(slide, "1:15", [
    "强调 idea 常常不是从“凭空创新”开始，而是从一个可重复观察到的失败开始。",
    "把两个现象分别归类：一个是模型层的问题（如何比较模态），一个是数据层的问题（如何增加稀有损伤暴露）。",
    "转场：接下来演示如何把这两句观察交给 AI，而不是问“给我一个新方法”。",
  ], [`${ROOT}/latex/main.tex（Introduction）`]);
}

// 06 — Detailed idea prompt
{
  const slide = buildSlide11(presentation, {
    title: title("案例 1｜先给“失败现象”，再让 AI 扩展假设"),
    body1: topicBlock("一个好提示词要包含 5 件事", "任务背景 · 已观察到的失败 · 不允许改变的约束 · 需要的对照实验 · 证伪条件"),
    body2: rt("低质量问题", 26, { bold: true, color: RED }),
    body3: rt("可研究的问题", 26, { bold: true, color: BLUE }),
    body4: { detailGoesHere: rt("“请给我 10 个创新点。”", 20, { color: MUTED }), detailGoesHere2: rt("输出看似丰富，却无法比较、验证，也容易重复已有工作。", 18, { color: MUTED }), detailGoesHere3: rt("", 18) },
    body5: { detailGoesHere: rt("“根据失败现象，给 3 个机制假设。”", 20, { color: MUTED }), detailGoesHere2: rt("每个假设必须给最小实现、对照、风险与证伪条件。", 18, { color: MUTED }), detailGoesHere3: rt("", 18) },
    footer1: footer(6),
  });
  addPromptBox(slide,
    "任务：灾前光学 + 灾后 SAR 的四类建筑损伤分割\n失败：简单拼接增益有限；受损/毁坏类别稀少\n约束：不重做解码器；推理开销尽量不变；必须可消融\n请提出 3 个可证伪机制假设。每个输出：\n① 为什么可能有效  ② 最小改动  ③ 必要对照  ④ 失败风险  ⑤ 证伪条件",
    { left: 42, top: 278, width: 1197, height: 255 },
  );
  notes(slide, "1:50", [
    "先读左侧低质量问题，让听众感受“创新点列表”为什么没有用：它没有锚定观察，也没有约束。",
    "逐行读可复制提示词。重点解释“最小改动”和“证伪条件”：前者控制工程成本，后者逼 AI 提出真正可以被实验否定的假设。",
    "实操建议：第一次只让 AI 产出 3 个假设；第二轮再要求它互相批评，避免一口气生成 20 个点子。",
  ], [`${ROOT}/latex/main.tex（problem formulation）`]);
}

// 07 — Hypothesis tree
{
  const slide = buildSlide18(presentation, {
    title: title("AI 的价值不是“给答案”，而是展开假设空间"),
    body1: block("模型层假设", "如果让灾前光学作为 Query，去检索灾后 SAR 的 Key/Value，就能显式比较结构变化。\n\n→ CMCA", 21),
    body2: block("数据层假设", "如果优先混入跨事件的受损/毁坏区域，并同步三元组，就能增加稀有损伤暴露。\n\n→ DA-CutMix", 21),
    body3: block("评估层假设", "如果增益来自更稳健的变化建模，它不应只在一个骨干或总体像素统计上成立。\n\n→ 多骨干 + 逐事件", 21),
    label1: rt("机制", 22, { bold: true, color: BLUE }),
    label2: rt("数据", 22, { bold: true, color: BLUE }),
    label3: rt("证据", 22, { bold: true, color: BLUE }),
    footer1: footer(7),
  });
  notes(slide, "1:40", [
    "按三列讲：机制、数据、证据。前两列是候选方法，第三列决定实验是否足以支持结论。",
    "提醒听众：AI 会把语言说得很肯定；我们要把每句话改写为“如果……那么在某个对照中应该观察到……”。",
    "可以让 AI 继续做红队：分别给出每个假设最可能失败的原因，例如形状不匹配、显存爆炸、混入不真实区域。",
  ], [
    `${ROOT}/latex/main.tex（CMCA and DA-CutMix descriptions）`,
    `${ROOT}/bda_benchmark/model/UNetCMCA.py`,
  ]);
}

// 08 — Falsifiable design with roadmap
{
  const slide = buildSlide08(presentation, {
    title: title("两个想法，两个独立插件"),
    body1: block("可证伪设计", "CMCA：只改语义瓶颈的跨模态交互。\nDA-CutMix：只改训练数据，不改推理。\n\n四组对照：Baseline / +CMCA / +DA-CutMix / +两者。\n\n失败也有信息：若组合不增益，检查是否互相冗余或依赖骨干。", 21),
    footer1: footer(8),
  });
  await addImage(slide, path.join(ROOT, "latex/images/Technology Roadmap.png"), { left: 658, top: 118, width: 581, height: 512 }, { alt: "CMCA and DA-CutMix technology roadmap", fit: "contain" });
  addCaseTag(slide, "IDEA → 设计 → 对照", 42, 148);
  notes(slide, "1:25", [
    "指着图从左到右讲：训练前先做 DA-CutMix，网络瓶颈处做 CMCA，最后保持原损失与预测头。",
    "为什么要做成插件？因为插件能在多个骨干上复用，也能让两个贡献独立开关，便于消融。",
    "这一步可以让 AI 帮你检查实验矩阵是否漏项，但矩阵对应什么科学问题必须由人决定。",
  ], [
    `${ROOT}/latex/images/Technology Roadmap.png`,
    `${ROOT}/latex/main.tex（Overall Framework）`,
  ]);
}

// 09 — Human screening
{
  const slide = buildSlide13(presentation, {
    title: title("留下 idea 前，我用四道门把“漂亮废话”筛掉"),
    body1: blockGoes("可解释", "能否说清它针对哪个失败机制，而不是只说“增强特征”？", 20),
    body2: blockGoes("可实现", "最小改动在哪里？shape、显存、数据同步是否可控？", 20),
    body3: blockGoes("可证伪", "有明确基线、消融、负面结果解释和停止条件吗？", 20),
    body4: blockGoes("可定位新意", "与最相近工作相比，差异是机制、设置还是证据？", 20),
    footer1: footer(9),
  });
  notes(slide, "1:10", [
    "建议把 AI 候选 idea 放进这四格，任何一格答不清就先不做。",
    "尤其警惕“可定位新意”：AI 很容易把常见模块重新命名。必须去原论文核对，不接受二手总结。",
    "在这个案例里，CMCA 的最小接口和 DA-CutMix 的同步三元组都能被清楚描述，因此进入下一步文献核查。",
  ]);
}

// 10 — Literature questions
{
  const slide = buildSlide13(presentation, {
    title: title("案例 2｜查文献之前，先写下四个问题"),
    body1: blockGoes("任务线", "建筑损伤评估的主流任务定义、数据集与指标是什么？", 20),
    body2: blockGoes("融合线", "光学—SAR 如何对齐、交互、比较？简单拼接哪里失效？", 20),
    body3: blockGoes("长尾线", "CutMix、ClassMix、Copy-Paste 如何增加稀有类别证据？", 20),
    body4: blockGoes("泛化线", "总体分数之外，跨事件/跨灾种如何划分和报告？", 20),
    footer1: footer(10),
  });
  addCaseTag(slide, "检索不是收集标题，而是回答问题", 42, 103);
  notes(slide, "1:15", [
    "演示如何从两个失败现象拆出四条检索线。每条线都对应论文里的一个论证任务。",
    "不要一开始用完整自然语言问“这个方向的 gap 是什么”；先分别找任务、机制、长尾和泛化证据。",
    "这样做的好处是：即使 AI 漏掉论文，你也能看出缺的是哪一类证据。",
  ], [`${ROOT}/latex/main.tex（Related Work）`]);
}

// 11 — Layered search
{
  const slide = buildSlide17(presentation, {
    title: title("检索分三层：先扩召回，再追原文，最后做反证"),
    label1: rt("第一层", 18, { bold: true, color: BLUE }),
    label2: rt("第二层", 18, { bold: true, color: BLUE }),
    label3: rt("第三层", 18, { bold: true, color: BLUE }),
    body1: block("广搜术语", "用 Semantic Scholar / Elicit / Perplexity 找同义词、代表作、引用链。\n输出：20 篇候选，不下结论。", 20),
    body2: block("核对原文", "下载 PDF，定位任务设置、数据划分、公式、表格与限制。\n输出：证据卡片。", 20),
    body3: block("反向搜索", "专门搜失败案例、负面结果、跨域退化和相反结论。\n输出：边界条件。", 20),
    footer1: footer(11),
  });
  notes(slide, "1:20", [
    "第一层允许 AI 快，但只把结果当候选目录；第二层必须回原文；第三层主动找反证。",
    "给一个可复制检索式示例：('building damage assessment' AND optical SAR) AND (cross-attention OR fusion)；再把 augmentation 和 cross-event 分开搜。",
    "实操小技巧：让 AI 为每个关键词给 3 个同义表达，但不要让它编造 DOI。",
  ]);
}

// 12 — Evidence card
{
  const slide = buildSlide11(presentation, {
    title: title("一篇论文只留一张“证据卡”，避免读完就忘"),
    body1: topicBlock("示例：BDANet 的证据卡", "AI 可以先抽取；研究者必须回到原文核对页码、表格和实验设置。"),
    body2: rt("证据卡内容", 26, { bold: true, color: BLUE }),
    body3: rt("给 AI 的输出契约", 26, { bold: true, color: BLUE }),
    body4: {
      detailGoesHere: rt("主张：类聚焦的 CutMix 有助于困难损伤等级", 18, { color: MUTED }),
      detailGoesHere2: rt("设置：光学 BDA；需记录数据、基线、指标", 18, { color: MUTED }),
      detailGoesHere3: rt("限制：未处理光学—SAR—标签三元同步", 18, { color: MUTED }),
    },
    body5: {
      detailGoesHere: rt("只输出原文明确支持的内容", 18, { color: MUTED }),
      detailGoesHere2: rt("每条附页码/表号/原句附近上下文", 18, { color: MUTED }),
      detailGoesHere3: rt("无法确认写“未找到”，禁止补全", 18, { color: MUTED }),
    },
    footer1: footer(12),
  });
  addPromptBox(slide, "请把这篇论文整理为：研究问题｜方法机制｜数据与划分｜关键对照｜数字证据｜作者承认的限制｜与我课题的关系。每条必须附页码或表号；无法确认就写“未找到”。", { left: 42, top: 285, width: 1197, height: 70 });
  notes(slide, "1:30", [
    "现场演示时先展示提示词，再讲左侧卡片。AI 的任务是结构化，不是替你判断论文好不好。",
    "核查顺序：先看表格数字是否存在，再看数据划分是否一致，最后看作者是否真的做了你想引用的结论。",
    "卡片最后一行必须写限制；没有限制的卡片通常只是摘要复述。",
  ], [
    `${ROOT}/latex/main.bbl（shen2022bdanet）`,
    `${ROOT}/latex/main.tex（Sample-Mixing Augmentation）`,
  ]);
}

// 13 — Gap matrix
{
  const slide = buildSlide14(presentation, {
    title: title("文献矩阵最终要产出“可防守的 gap”"),
    body1: topicBlock("不是：没人做过。  而是：现有机制在这个设置、这个证据要求下还缺什么。", "把相近工作放在同一行比较，gap 会从形容词变成明确的设计决策。"),
    footer1: footer(13),
  });
  coverExistingRegion(slide, { left: 35, top: 220, width: 1210, height: 420 });
  const values = [
    ["研究线", "已有工作", "剩余缺口", "本项目决策"],
    ["跨模态融合", "拼接 / 对齐 / 通用注意力", "缺少以灾前结构为查询的显式变化比较", "CMCA：Optical Q，SAR K/V"],
    ["稀有类别", "CutMix / ClassMix / Copy-Paste", "未同时保证跨事件与三元组同步", "DA-CutMix：跨事件、损伤富集"],
    ["泛化证据", "总体测试集指标", "平均分可能掩盖事件级退化", "报告整体 + 逐事件 mIoU"],
    ["可迁移性", "为单一骨干定制", "难区分模块收益与骨干收益", "6 个不同家族骨干受控消融"],
  ];
  const table = slide.tables.add({ rows: 5, columns: 4, left: 42, top: 236, width: 1197, height: 385, columnWidths: [175, 300, 365, 357], values });
  table.borders.assign({ style: "solid", fill: RULE, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 4 }).assign({ fill: "#111827", textStyle: { fontSize: 19, bold: true, color: "#FFFFFF", typeface: FONT }, anchor: "middle" });
  table.cells.block({ row: 1, column: 0, rowCount: 4, columnCount: 4 }).assign({ fill: "#FFFFFF", textStyle: { fontSize: 17, color: INK, typeface: FONT }, anchor: "middle" });
  table.cells.block({ row: 1, column: 3, rowCount: 4, columnCount: 1 }).assign({ fill: "#EDF5FF", textStyle: { fontSize: 17, bold: true, color: "#134B9A", typeface: FONT }, anchor: "middle" });
  coverExistingRegion(slide, { left: 40, top: 625, width: 1200, height: 30 });
  notes(slide, "1:35", [
    "逐行解释“已有工作—剩余缺口—项目决策”，让听众看到方法不是从工具名跳出来的，而是从证据矩阵推出来的。",
    "强调最安全的创新表述不是“first”，而是明确限定任务、设置和证据：在多模态 BDA 的三元同步与跨事件评估下做了什么。",
    "如果某行找不到直接证据，就回到检索层补文献，而不是让 AI 用常识填满。",
  ], [
    `${ROOT}/latex/main.tex（Related Work and Contributions）`,
    `${ROOT}/latex/main.bbl`,
  ]);
}

// 14 — Coding details
{
  const slide = buildSlide05(presentation, {
    title: title("案例 3｜让 AI 改“最小接口”，不要重写整个项目"),
    body1: block("", "", 18),
    body2: block("", "", 18),
    footer1: footer(14),
  });
  coverExistingRegion(slide, { left: 35, top: 175, width: 1210, height: 450 });
  addCodeBox(slide, "CMCA：只替换瓶颈交互", `# Optical asks; SAR answers\nQ = q_proj(opt_feat)\nK = k_proj(reduce(sar_feat))\nV = v_proj(reduce(sar_feat))\nattn = softmax(Q @ K.T * scale)\naligned_sar = attn @ V\nchange = change_proj([aligned_sar, opt_feat])`, { left: 42, top: 205, width: 572, height: 320 }, CYAN);
  addCodeBox(slide, "DA-CutMix：只改训练数据", `donor = sample(different_event)\nbox = find_patch(\n  damage_pixels >= threshold\n)\npre[box]   = donor_pre[box]\npost[box]  = donor_post[box]\nlabel[box] = donor_label[box]\n# inference path unchanged`, { left: 666, top: 205, width: 572, height: 320 }, CYAN);
  addTextBox(slide, "我给编程 AI 的要求：先列接口与 shape → 只生成最小补丁 → 补一个最小张量测试 → 显存/随机性/推理路径自检", { left: 42, top: 553, width: 1197, height: 58 }, { fontSize: 21, bold: true, color: BLUE, fill: "#EDF5FF", line: { style: "solid", fill: PALE_BLUE, width: 1 }, borderRadius: 12, verticalAlignment: "middle", insets: { top: 8, right: 14, bottom: 8, left: 14 } });
  addCaseTag(slide, "代码实践 · 接口优先", 42, 150);
  notes(slide, "1:55", [
    "先展示左右两个最小接口：模型侧输入输出 shape 不变；数据侧同步替换 pre/post/label，推理路径不变。",
    "可复制的编程提示词：‘先阅读相关类和调用链；列出最小插入点与张量形状；不要改公共 API；提交补丁后运行一个 B=2 的随机张量前向测试。’",
    "人工重点检查三件事：跨模态方向是否与科学假设一致；数据增强有没有破坏标签对齐；随机增强是否记录概率与成功率。",
    "如果 AI 一次改 20 个文件，立即缩小任务；科研代码最怕无法定位增益来源。",
  ], [
    `${ROOT}/bda_benchmark/model/UNetCMCA.py`,
    `${ROOT}/bda_benchmark/dataset/imutils.py`,
    `${ROOT}/bda_benchmark/dataset/make_data_loader.py`,
  ]);
}

// 15 — Experiment design
{
  const slide = buildSlide18(presentation, {
    title: title("每个科研主张，都必须对应一个清楚的比较"),
    body1: block("2 × 2 消融", "Baseline\n+ CMCA\n+ DA-CutMix\n+ 两者\n\n回答：两个模块独立还是互补？", 20),
    body2: block("6 类骨干", "CNN · Transformer\nSiamese · Recurrent\nDeepLab · Mamba\n\n回答：收益是否依赖单一架构？", 20),
    body3: block("两种统计口径", "整体测试集 mIoU\n逐事件平均 / 胜出次数\n\n回答：增益是真稳健，还是大事件像素主导？", 20),
    label1: rt("模块因果", 20, { bold: true, color: BLUE }),
    label2: rt("跨架构", 20, { bold: true, color: BLUE }),
    label3: rt("跨事件", 20, { bold: true, color: BLUE }),
    footer1: footer(15),
  });
  notes(slide, "1:25", [
    "把论文的三个主要主张一一映射到三类实验；如果一个主张找不到对应比较，就删掉或补实验。",
    "AI 可以根据主张生成实验清单，但要让它同时标注：自变量、控制变量、主要指标、停止条件和预期反例。",
    "本案例刻意使用六种骨干，并同时看全局像素与逐事件结果，避免只挑一个好看的数字。",
  ], [`${ROOT}/latex/main.tex（Experimental Setup and Ablations）`]);
}

// 16 — Results chart
{
  const slide = buildSlide21(presentation, {
    title: title("结果不是一句“提升了”：总体与逐事件可能讲不同故事"),
    body1: rt("6 个骨干的总体 mIoU：Baseline vs. CMCA + DA-CutMix", 24, { bold: true }),
    stat1: rt("6 / 6", 44, { bold: true, color: BLUE }),
    stat2: rt("+1.94", 44, { bold: true, color: BLUE }),
    body2: rt("组合设置的总体 mIoU 均高于基线", 18, { color: MUTED }),
    body3: rt("DeepLabV3Plus 的最大增益（百分点）", 18, { color: MUTED }),
    footer1: footer(16),
  });
  coverExistingRegion(slide, { left: 35, top: 122, width: 596, height: 545 });
  slide.charts.add("bar", {
    position: { left: 48, top: 155, width: 565, height: 470 },
    categories: ["UNet", "DamageFormer", "SiamAttn", "SiamCRNN", "DeepLab", "ChangeMamba"],
    series: [
      { name: "Baseline", values: [64.74, 66.84, 65.03, 66.13, 63.85, 66.73], fill: "#C9CED6" },
      { name: "+ CMCA + DA-CutMix", values: [66.16, 67.14, 66.28, 66.52, 65.79, 67.43], fill: BLUE },
    ],
    barOptions: { direction: "column", grouping: "clustered", gapWidth: 60 },
    hasLegend: true,
    legend: { position: "bottom", overlay: false, textStyle: { fontSize: 13, fill: MUTED } },
    yAxis: { min: 63, max: 68, majorUnit: 1, title: "mIoU (%)", textStyle: { fontSize: 12, fill: MUTED }, majorGridlines: { style: "solid", fill: "#E5E7EB", width: 1 } },
    xAxis: { textStyle: { fontSize: 11, fill: MUTED }, line: { style: "solid", fill: RULE, width: 1 } },
    dataLabels: { showValue: true, position: "outEnd", textStyle: { fontSize: 11, fill: INK, bold: true } },
    chartFill: "#FFFFFF",
    chartLine: { style: "solid", fill: "#FFFFFF", width: 0 },
    plotAreaFill: { type: "none" },
    plotAreaLine: { style: "solid", fill: "#FFFFFF", width: 0 },
  });
  addTextBox(slide, "关键解释\nDamageFormer：全局最优来自 DA-CutMix 单独使用；某些设置的逐事件平均更稳。\n\n结论应写成“收益受骨干和统计口径影响”，而不是“所有场景一致提升”。", { left: 658, top: 164, width: 581, height: 160 }, { fontSize: 20, color: MUTED, fill: "#F7F8FA", line: { style: "solid", fill: RULE, width: 1 }, borderRadius: 12, insets: { top: 14, right: 16, bottom: 14, left: 16 } });
  notes(slide, "1:40", [
    "先讲蓝柱整体高于灰柱，再立刻补充例外：组合并不总是每个骨干的单项最优，逐事件平均也可能给出不同排序。",
    "这是 AI 很容易写错的地方：它会把“多数提升”扩写成“稳定提升/显著提升”。必须逐组对照原表。",
    "强调统计口径的意义：全局 mIoU 按所有像素汇总，逐事件平均让小事件获得同等权重。",
  ], [
    `${ROOT}/latex/main.tex（Table: Overall Ablation Results; Per-Event Robustness）`,
    `${ROOT}/Ablation experiment.xlsx`,
  ]);
}

// 17 — Visualization questions
{
  const slide = buildSlide10(presentation, {
    title: title("案例 4｜画图前先写一句“观众必须看见什么”"),
    body1: rt("本页的句子：组合模块在所有骨干上提高总体 mIoU，但收益大小与逐事件稳定性因骨干而异。", 25, { bold: true }),
    label1: rt("① 主要比较是什么？", 20, { bold: true, color: BLUE }),
    label2: rt("② 基线和单位是什么？", 20, { bold: true, color: BLUE }),
    label3: rt("③ 需要显示绝对值还是 Δ？", 20, { bold: true, color: BLUE }),
    label4: rt("④ 异常与反例在哪里？", 20, { bold: true, color: BLUE }),
    label5: rt("⑤ 一张图能否回答一个问题？", 20, { bold: true, color: BLUE }),
    body2: {
      loremIpsumDolorSitAmetConsecteturAdipiscing: rt("AI 可以写绘图代码、统一字体、输出 SVG/PDF；但图表类型必须由信息任务决定。", 21, { color: MUTED }),
      loremIpsumDolorSitAmetConsecteturAdipiscing2: rt("如果一句话写不清，先不要画图。", 21, { bold: true, color: BLUE }),
    },
    footer1: footer(17),
  });
  notes(slide, "1:10", [
    "把上一页的结论句读出来，说明图表只是这句话的证据载体。",
    "实操提示词：把表格列名和目标句给 AI，让它先推荐 2 种图并解释各自会隐藏什么，再让它生成代码。",
    "检查三件事：纵轴是否截断、颜色是否把次要信息抢走、图例是否让人来回寻找。",
  ], [`${ROOT}/latex/main.tex（reported metrics and result interpretation）`]);
}

// 18 — Spreadsheet to evidence
{
  const slide = buildSlide08(presentation, {
    title: title("原始表格 → 证据图"),
    body1: block("1 选指标", "围绕主张只保留 mIoU。\n\n2 算差值", 21),
    footer1: footer(18),
  });
  coverExistingRegion(slide, { left: 41, top: 205, width: 581, height: 420 });
  addTextBox(slide, "1  选指标\n围绕主张只保留 mIoU，而不是一次放 8 个指标。\n\n2  算差值\n保留 Baseline 与组合设置，计算百分点变化。\n\n3  留例外\n把 DamageFormer 和逐事件口径的差异写进解释，不藏掉。", { left: 42, top: 214, width: 550, height: 370 }, { fontSize: 22, color: MUTED });
  await addImage(slide, path.join(ROOT, "tmp/spreadsheet-read/renders/Overall.png"), { left: 635, top: 120, width: 604, height: 510 }, { alt: "Ablation workbook overall results table", fit: "contain" });
  addCaseTag(slide, "原始数据保留在表格；幻灯片只展示当前主张所需证据", 42, 145);
  notes(slide, "1:20", [
    "右侧是原始工作表视图，信息完整但不适合讲述；左侧展示三次压缩：选指标、算差值、留例外。",
    "让 AI 生成绘图脚本时，要求它同时输出数据核对表：原始单元格、转换公式、图中数值。这样能防止复制错列。",
    "科研画图不等于美化：每一次删除都要说明为什么不影响当前主张。",
  ], [
    `${ROOT}/Ablation experiment.xlsx`,
    `${ROOT}/tmp/spreadsheet-read/renders/Overall.png`,
  ]);
}

// 19 — Claim evidence warrant
{
  const slide = buildSlide17(presentation, {
    title: title("案例 5｜论文段落用“主张—证据—解释”搭起来"),
    label1: rt("CLAIM", 18, { bold: true, color: BLUE }),
    label2: rt("EVIDENCE", 18, { bold: true, color: BLUE }),
    label3: rt("WARRANT", 18, { bold: true, color: BLUE }),
    body1: block("主张", "CMCA 让跨模态比较更显式，有助于恢复损伤敏感结构线索。", 20),
    body2: block("证据", "UNet：受损类 IoU 37.48 → 39.30；mIoU 64.74 → 65.39。", 20),
    body3: block("解释", "提升集中在受损类，支持“比较结构变化”的机制；但仍需用多骨干与事件级结果限定外推。", 20),
    footer1: footer(19),
  });
  notes(slide, "1:15", [
    "让听众看到一段结果分析不是‘数字 + 形容词’，而是三步：我主张什么、什么数字支持、为什么这个数字与机制有关。",
    "可复制提示词：‘按 claim-evidence-warrant 重写，不新增任何数字；每个形容词必须能指向表格单元格。’",
    "AI 最适合做结构检查：标出只有主张没有证据、只有数字没有解释的句子。",
  ], [`${ROOT}/latex/main.tex（UNet ablation discussion）`]);
}

// 20 — Before/after writing
{
  const slide = buildSlide05(presentation, {
    title: title("AI 润色的目标：更精确，不是更夸张"),
    body1: block("改写前：看起来有力，但不可核验", "“Our method significantly improves performance and generalization across all settings.”\n\n问题：没有数字；‘significantly’ 暗示统计检验；‘all settings’ 与结果不符。", 21),
    body2: block("改写后：数字、范围、例外都在", "“On UNet, CMCA + DA-CutMix raises mIoU from 64.74% to 66.16% (+1.42 points). The gain is backbone-dependent: for DamageFormer, DA-CutMix alone gives the best global mIoU.”", 20),
    footer1: footer(20),
  });
  slide.shapes.add({ geometry: "rect", position: { left: 41, top: 594, width: 581, height: 8 }, fill: RED, line: { style: "solid", fill: RED, width: 0 } });
  slide.shapes.add({ geometry: "rect", position: { left: 657, top: 594, width: 581, height: 8 }, fill: BLUE, line: { style: "solid", fill: BLUE, width: 0 } });
  notes(slide, "1:30", [
    "先让听众挑错：significantly、generalization、all settings 分别需要什么证据。",
    "再读右侧：它只使用表格已有数字，并主动交代 DamageFormer 的例外，因此结论更窄但更可信。",
    "实操流程：先让 AI 做‘claim audit’，列出每句主张需要的证据；再润色语言；最后人工逐个数字回表格。顺序不要反。",
  ], [`${ROOT}/latex/main.tex（Overall Ablation Results discussion）`]);
}

// 21 — Reliability boundary
{
  const slide = buildSlide13(presentation, {
    title: title("接受 AI 输出前，我固定做四类检查"),
    body1: blockGoes("引用检查", "论文是否真实？作者、年份、页码、表号是否与原文一致？", 20),
    body2: blockGoes("数据与代码检查", "数字回到原表；代码能跑测试；随机种子、环境与日志可复现。", 20),
    body3: blockGoes("隐私与权限检查", "未公开数据、审稿材料、个人信息和密钥不直接上传公共模型。", 20),
    body4: blockGoes("作者责任检查", "AI 不署名也不背责；最终论证、引用和表述由作者确认。", 20),
    footer1: footer(21),
  });
  notes(slide, "1:15", [
    "这页不要一带而过。举一个最常见错误：AI 给出一个看似真实的引用，但年份或结论对应不上。",
    "建议建立简单审计记录：关键提示词、输入文件版本、代码提交、生成图表脚本、人工核查者。",
    "如果材料涉及未公开数据或审稿保密，先确认机构和期刊政策，再选择本地/企业环境或完全不上传。",
  ]);
}

// 22 — Reusable prompt contract
{
  const slide = buildSlide15(presentation, {
    title: title("我反复复用的不是“神奇提示词”，而是一份任务合同"),
    body1: {
      titleHere: rt("复制这个骨架", 28, { bold: true, spaceAfter: 800 }),
      loremIpsumDolorSitAmetConsecteturAdipiscing: rt("背景：我在做什么，听众/读者是谁。\n任务：这一次只完成一个动作。\n约束：哪些不能改，哪些证据不能越界。\n输入：文件、表格、段落或代码。\n输出：指定格式、长度和字段。\n自检：列出不确定项、反例和验证步骤。", 20, { color: MUTED }),
      quamUtMassaLuctusCursusNullamPharetra: rt("", 18),
    },
    body2: rt("上下文 + 单一任务", 19, { bold: true, color: BLUE }),
    body3: rt("约束 + 证据边界", 19, { bold: true, color: BLUE }),
    body4: rt("输出格式 + 样例", 19, { bold: true, color: BLUE }),
    body5: rt("自检 + 失败条件", 19, { bold: true, color: BLUE }),
    label1: rt("", 16), label2: rt("", 16), label3: rt("", 16), label4: rt("", 16), label5: rt("", 16),
    footer1: footer(22),
  });
  notes(slide, "1:20", [
    "把它解释成任务合同：告诉 AI 做什么，也告诉它什么情况下应该停下并说不知道。",
    "示范一句自检要求：‘输出后列出三项最可能错误的地方，以及我应该打开哪个文件、哪张表来验证。’",
    "对于复杂任务，用两轮：第一轮只让 AI 复述理解与计划；确认后第二轮才生成结果。",
  ]);
}

// 23 — Close
{
  const slide = buildSlide26(presentation, {
    title: rt("带走一句话", 24, { bold: true, color: BLUE }),
    title2: rt("AI 把科研循环变快\n人把它变成科学", 76, { bold: true, lineSpacingPercent: 93000 }),
    title3: {
      loremIpsumDetails: rt("明天就做：选一个最卡的步骤", 23, { bold: true, spaceAfter: 400 }),
      loremIpsumDetails2: rt("用“任务合同”跑 3 轮", 23, { spaceAfter: 400 }),
      loremIpsumDetails3: rt("逐条验证后，只保留有证据的结果", 23),
    },
  });
  slide.shapes.add({ geometry: "rect", position: { left: 1038, top: 545, width: 200, height: 12 }, fill: BLUE, line: { style: "solid", fill: BLUE, width: 0 } });
  notes(slide, "0:50", [
    "回扣开场：AI 的真正价值是缩短观察—假设—实现—验证—表达的循环。",
    "给听众一个具体行动：不要一次改造整个工作流；只挑当前最卡的一步，按任务合同跑三轮，并留下验证记录。",
    "结束时邀请提问：你们可以把自己的研究卡点说出来，我们现场把它改写成一个可验证的 AI 任务。",
  ]);
}

await fs.mkdir(RENDER_DIR, { recursive: true });

for (const [index, slide] of presentation.slides.items.entries()) {
  const stem = `slide-${String(index + 1).padStart(2, "0")}`;
  const png = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(RENDER_DIR, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: "layout" });
  await fs.writeFile(path.join(RENDER_DIR, `${stem}.layout.json`), await layout.text());
}

const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
await fs.writeFile(path.join(TMP, "deck-montage.webp"), new Uint8Array(await montage.arrayBuffer()));

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(OUT);

console.log(`Saved ${OUT}`);
