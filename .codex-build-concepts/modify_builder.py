from pathlib import Path
s=Path('/private/tmp/cmca_technical_render/build_artifacts/build.mjs').read_text()
s=s.replace("'/Users/haochen/.codex/plugins/cache/openai-primary-runtime/presentations/26.1015.12040/skills/presentations'","'/Users/haochen/.codex/plugins/cache/openai-primary-runtime/presentations/26.1020.11127/skills/presentations'")
s=s.replace("'.codex-build-technical'","'.codex-build-concepts'").replace('组会汇报_202610_陈昊_CMCA_DACutMix_技术讲解与符号详释版.pptx','组会汇报_202610_陈昊_CMCA_DACutMix_特征图与编解码说明_无页码版.pptx')
s=s.replace("  txt(slide,String(ppt.slides.items.length).padStart(2,'0'),1158,682,60,17,12,C.muted,false,{align:'right'});\n",'')
s=s.replace("  txt(s,String(ppt.slides.items.length).padStart(2,'0'),1156,672,62,22,14,C.white,false,{align:'right'});\n",'')
s=s.replace(" txt(s,`${ppt.slides.items.length} / ${ppt.slides.items.length}`,1120,675,95,20,14,'#B8C4CC',false,{align:'right'});\n",'')
insert='''// Concepts before the encoder equations.
{
 const s=ppt.slides.add();addHeader(s,'特征图：空间网格上的多通道表示','连接原始影像与后续注意力计算的中间数据');
 asset(s,'sym_fl.png',86,157,1090,77,'第 l 层特征图的张量形状');
 table(s,[['维度','表示什么','本研究中的含义'],
   ['hₗ × wₗ','特征图的空间位置数','下采样后的二维网格，每格对应输入中的一个局部区域'],
   ['Cₗ','每个位置的通道数','每个位置是一组 Cₗ 维数值，由编码器学习得到'],
   ['m ∈ {o, s}','特征来自哪种影像','o 为灾前光学，s 为灾后 SAR；两支分别编码']
 ],76,279,1129,259,[210,320,599],20);
 txt(s,'例：640 × 640 输入，经 UNet 深层编码得到 512 × 40 × 40 的特征图。',83,568,1110,40,25,C.navy,true);
 txt(s,'40 × 40 是空间网格；每个网格位置有 512 个特征值，不是 512 个类别。',83,620,1110,44,22,C.ink);
 note(s,'承接论文 III-C 与式（2）。特征图是 C×h×w 的数值张量，不是人工可直接辨认的彩色图。每个空间位置有 C 个通道，编码器通过训练形成这些表示。空间尺寸会随网络下采样而变小；示例 UNet 的 512×40×40 来自论文 Table IV，640×640 是训练裁剪尺寸。不要把一个特征位置说成必然对应一栋建筑，其感受范围与编码器结构有关。');
}
{
 const s=ppt.slides.add();addHeader(s,'Encoder 与 Decoder：提取特征并恢复像素级预测','以双分支编码器与 UNet 式解码路径说明');
 table(s,[['阶段','输入 → 输出','承担的工作'],
   ['Encoder（编码器）','光学、SAR 各自输入 → 多层特征图','逐层提取结构与灾后响应特征；空间尺度逐渐缩小'],
   ['CMCA 与适配器','成对深层特征 → 融合特征','光学 Q 查询 SAR K/V，再匹配原网络接口'],
   ['Decoder（解码器）','深层融合特征 → 原图尺度','逐步恢复空间分辨率，结合可用的浅层 skip 特征'],
   ['预测头','原图尺度特征 → H × W × 4','每个像素得到四个类别得分，选出预测标签']
 ],73,170,1135,367,[230,350,555],20);
 txt(s,'Encoder 回答“提取了什么线索”；Decoder 回答“这些线索对应图上的哪些像素”。',81,570,1100,62,24,C.navy,true);
 note(s,'承接论文 Figure 1、III-B、III-C。Encoder 是一系列可学习的编码阶段 E_l^m，分别处理光学与 SAR；其输出为多尺度特征图。CMCA 在选定深层交互，输出适配器连接到原解码器。Decoder 使用深层变化信息，并在带 skip 的骨干中接收浅层空间细节，逐步输出原图尺寸的四类预测。H×W×4 指每个像素有四个类别得分，最终类别图的形状为 H×W。不同骨干的具体解码结构不同，此页以 UNet 式路径示意。');
}
'''
needle="{\n const s=ppt.slides.add();addHeader(s,'双分支编码：逐层建立模态特定表征'"
assert needle in s
s=s.replace(needle,insert+needle)
needle="execFileSync(py,[path.join(buildDir,'patch_fonts.py'),dedupPath,fontPath]);"
s=s.replace(needle,needle+"\nconst mergedPath=path.join(buildDir,'candidate_with_current_edits.pptx');\nexecFileSync(py,[path.join(buildDir,'transplant_current.py'),path.join(outputDir,'组会汇报_202610_陈昊_CMCA_DACutMix_技术讲解与符号详释版.pptx'),fontPath,mergedPath]);")
s=s.replace('workspaceDir,candidatePath:fontPath,finalPath,pythonExecutable:py,','workspaceDir,candidatePath:mergedPath,finalPath,pythonExecutable:py,')
Path('.codex-build-concepts/build.mjs').write_text(s)
